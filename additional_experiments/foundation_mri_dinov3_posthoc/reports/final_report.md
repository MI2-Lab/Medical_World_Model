# DINOv3 DCE-MRI post-hoc 敏感性分析

## 结论边界

本报告只增加一个固定模型：`dinov3_vitb16_lvd1689m_posthoc`。它是在原 foundation-MRI 正式结果已经公开后由用户指定的 **post-hoc（事后）敏感性 baseline**，不是原预注册候选，也不构成新的确认性检验。无论本报告中的数值如何，原实验的候选集合、主要分析和既有结论均保持不变；这里的结果只能作为后续 encoder 研究的描述性证据。

DINOv3 使用 Meta 的 custom license，而不是标准开源许可证；本地实验依赖本机既有、经 hash 验证的官方 checkpoint。本实验不代表机构或 PI 已接受许可，责任人仍需确认适用条款，报告与仓库不分发权重。许可措辞的 outcome-blind 澄清见 `reports/license_scope_clarification.md`。LVD-1689M 来自不可逐项枚举的公开网络图像池，没有 patient-level 成员清单或 I-SPY 排除清单，因此 I-SPY 衍生内容的预训练污染状态为 **unknown**，不能声称为零污染。

所有候选、GLOBAL/LOCAL 轴、T0/T0-T1/T0-T2 时点以及 full/complete-case 人群均按冻结协议顺序显示；没有 best-cell filtering，也没有按 AUROC、AUPRC、Brier 或区间排序。84 个比较、252 个 metric rows 均为同患者 outer-fold OOF 配对 bootstrap 的描述性结果（seed 2026、5000 次、percentile 95% CI），不用于模型/检查点选择或确认性显著性判断。差值均为“候选减参照”；AUROC/AUPRC 越大越好，Brier 越小越好。

## 模型与公平 adapter

- 官方 checkpoint：`facebook/dinov3-vitb16-pretrain-lvd1689m`，ViT-B/16；选择 B/16 是为了与原 DINO v1 ViT-B/16 reference 尽量公平地隔离预训练代际差异。
- 输入 adapter 与原 DINO v1 baseline 一致：DCE early、late、late-minus-pre 三通道，32 个 axial slices，224×224，ImageNet normalization，并保留相同 GLOBAL 与固定 central 64-mm LOCAL 几何。
- 每 slice 的 201 个 final tokens 按固定索引处理：CLS 为 `[0]`；四个 register tokens 为 `[1:5]` 并明确排除；196 个 patch tokens 为 `[5:201]`。表示为 `CLS[0]` 与 patch mean `[5:201]` 拼接，得到 1536 维；不把 register token 混入 patch pooling。

## 覆盖与可重复性

- full cohort：808
- radiomics complete-case：375
- DINOv3 pCR pooled cells：36
- 配对比较 specs：84
- 配对 metric rows：252
- outcome-blind comparison contract SHA-256：`f59533ebefcfb8fc48298386d4a9aa9d21bbeea5781e441e49b2210ec72e3760`
- 输出顺序为协议顺序，不是结果顺序。

## 12 个科学问题的读取方式

1. 使用模型：本扩展只使用上述 DINOv3 ViT-B/16；原实验模型不被替换。
2. 选择原因：与 DINO v1 的架构尺度、patch size 和 pooling adapter 对齐，同时明确 post-hoc、custom-license 与污染未知边界。
3. MRI-only AUROC/AUPRC：见下方 pCR 全量表的 `mri_only` 与 `mri_only_paired` 行，所有时点和空间轴均保留。
4. LOCAL vs GLOBAL：见比较族 `local_vs_global` 的全部 18 specs。
5. Foundation vs current CNN：见 `dinov3_vs_current_cnn_full` 的全部 12 个 full-cohort matched-axis specs。
6. HR/HER2 与 subtype decodability：见 phenotype 与 subtype 全量 matched 表；每格同时给 DINO v1、DINOv3 和 v3−v1 描述性差值。
7. FTV/ΔFTV decodability：见 FTV 全量 matched 表，static 与 delta endpoints 全部保留。
8. Clinical + Foundation 是否超过 clinical-only：见 `clinical_gain` 的 full 与 paired 全部 12 specs。
9. Clinical + FTV + Foundation 是否超过 clinical + FTV：见 `beyond_ftv` 的全部 6 specs。
10. 是否包含 tumor-size 以外信息：只能结合 `beyond_ftv`、FTV probe 和区间作描述性判断，不能由单个最佳 cell 推断。
11. Current World Model 是否 underuse MRI：只能结合 `dinov3_vs_current_cnn_full` 全部结果评估；本 post-hoc 分析不回写原结论。
12. 是否值得替换/增强 encoder：本报告可形成后续实验假设，但 custom license、污染未知和 post-hoc 性质要求独立预注册复验后再作工程决策。

## pCR pooled outer-fold OOF：全部 36 个 DINOv3 cells

| 模型 variant | 空间 | 时点 | 人群 | n | AUROC | AUPRC | Brier | 校准斜率 | ECE10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | GLOBAL | T0 | full_808 | 808 | 0.528 | 0.362 | 0.268 | 0.056 | 0.174 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | LOCAL | T0 | full_808 | 808 | 0.531 | 0.379 | 0.309 | 1.000 | 0.243 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | GLOBAL | T0 | full_808 | 808 | 0.704 | 0.549 | 0.222 | 0.905 | 0.148 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | LOCAL | T0 | full_808 | 808 | 0.693 | 0.527 | 0.229 | 1.160 | 0.169 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | GLOBAL | T0 | radiomics_complete_case_375 | 375 | 0.540 | 0.328 | 0.255 | 0.269 | 0.202 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | LOCAL | T0 | radiomics_complete_case_375 | 375 | 0.553 | 0.323 | 0.276 | 2.951 | 0.215 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | GLOBAL | T0 | radiomics_complete_case_375 | 375 | 0.601 | 0.386 | 0.280 | 1.000 | 0.258 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | LOCAL | T0 | radiomics_complete_case_375 | 375 | 0.611 | 0.381 | 0.268 | 2.643 | 0.236 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | GLOBAL | T0 | radiomics_complete_case_375 | 375 | 0.541 | 0.329 | 0.255 | 0.275 | 0.202 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | LOCAL | T0 | radiomics_complete_case_375 | 375 | 0.551 | 0.323 | 0.281 | 1.370 | 0.223 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | GLOBAL | T0 | radiomics_complete_case_375 | 375 | 0.601 | 0.384 | 0.277 | 1.604 | 0.257 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | LOCAL | T0 | radiomics_complete_case_375 | 375 | 0.611 | 0.381 | 0.268 | 2.644 | 0.236 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | GLOBAL | T0-T1 | full_808 | 808 | 0.501 | 0.351 | 0.333 | 1.000 | 0.279 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | LOCAL | T0-T1 | full_808 | 808 | 0.605 | 0.436 | 0.288 | 2.394 | 0.222 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | GLOBAL | T0-T1 | full_808 | 808 | 0.633 | 0.456 | 0.260 | 2.802 | 0.194 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | LOCAL | T0-T1 | full_808 | 808 | 0.657 | 0.469 | 0.245 | 2.940 | 0.170 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | GLOBAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.526 | 0.345 | 0.289 | 3.156 | 0.239 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | LOCAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.588 | 0.391 | 0.284 | 1.520 | 0.242 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | GLOBAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.563 | 0.364 | 0.286 | 1.241 | 0.236 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | LOCAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.621 | 0.393 | 0.262 | 3.368 | 0.219 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | GLOBAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.533 | 0.352 | 0.291 | 2.737 | 0.251 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | LOCAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.606 | 0.394 | 0.271 | 3.262 | 0.222 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | GLOBAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.579 | 0.376 | 0.276 | 2.371 | 0.219 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | LOCAL | T0-T1 | radiomics_complete_case_375 | 375 | 0.622 | 0.400 | 0.259 | 3.746 | 0.214 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | GLOBAL | T0-T2 | full_808 | 808 | 0.551 | 0.383 | 0.350 | 1.000 | 0.313 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only | LOCAL | T0-T2 | full_808 | 808 | 0.590 | 0.421 | 0.298 | 1.000 | 0.253 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | GLOBAL | T0-T2 | full_808 | 808 | 0.682 | 0.527 | 0.234 | 0.357 | 0.174 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical | LOCAL | T0-T2 | full_808 | 808 | 0.637 | 0.462 | 0.261 | 2.409 | 0.196 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | GLOBAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.484 | 0.286 | 0.319 | 1.000 | 0.299 |
| dinov3_vitb16_lvd1689m_posthoc_mri_only_paired | LOCAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.600 | 0.383 | 0.275 | 1.000 | 0.232 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | GLOBAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.539 | 0.324 | 0.324 | 1.000 | 0.285 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired | LOCAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.637 | 0.429 | 0.260 | 1.667 | 0.223 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | GLOBAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.484 | 0.287 | 0.312 | 1.000 | 0.290 |
| dinov3_vitb16_lvd1689m_posthoc_mri_ftv | LOCAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.602 | 0.384 | 0.274 | 1.000 | 0.242 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | GLOBAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.547 | 0.329 | 0.313 | 1.000 | 0.281 |
| dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv | LOCAL | T0-T2 | radiomics_complete_case_375 | 375 | 0.640 | 0.431 | 0.260 | 1.288 | 0.226 |

## 全部 84 个配对比较（252 个 metric rows）

| 比较族 | 人群 | 时点 | 参照 | 候选 | n | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔBrier [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dinov3_vs_dinov1 | full_808 | T0 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | -0.008 [-0.054, 0.038] | -0.018 [-0.057, 0.024] | -0.012 [-0.027, 0.002] |
| dinov3_vs_dinov1 | full_808 | T0 | dino_vitb16_imagenet1k_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.003 [-0.047, 0.054] | 0.005 [-0.045, 0.055] | 0.005 [-0.020, 0.030] |
| dinov3_vs_dinov1 | full_808 | T0 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | 0.001 [-0.026, 0.026] | 0.020 [-0.024, 0.062] | -0.006 [-0.013, 0.002] |
| dinov3_vs_dinov1 | full_808 | T0 | dino_vitb16_imagenet1k_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | 0.024 [-0.004, 0.053] | 0.029 [-0.013, 0.075] | -0.007 [-0.017, 0.002] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | 375 | 0.024 [-0.048, 0.096] | 0.010 [-0.045, 0.069] | -0.019 [-0.042, 0.002] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.013 [-0.059, 0.088] | -0.009 [-0.076, 0.061] | 0.004 [-0.025, 0.033] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | 0.013 [-0.043, 0.071] | -0.010 [-0.079, 0.069] | 0.010 [-0.016, 0.036] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | 0.006 [-0.054, 0.063] | -0.007 [-0.081, 0.056] | 0.007 [-0.018, 0.033] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | 375 | 0.025 [-0.046, 0.098] | 0.012 [-0.044, 0.071] | -0.019 [-0.042, 0.002] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.011 [-0.062, 0.085] | -0.009 [-0.077, 0.062] | 0.009 [-0.021, 0.039] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | 0.013 [-0.043, 0.071] | -0.012 [-0.082, 0.067] | 0.009 [-0.018, 0.035] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | 0.005 [-0.054, 0.063] | -0.007 [-0.081, 0.056] | 0.007 [-0.018, 0.033] |
| dinov3_vs_dinov1 | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | -0.049 [-0.096, -0.002] | -0.040 [-0.085, 0.005] | 0.044 [0.020, 0.069] |
| dinov3_vs_dinov1 | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.048 [-0.000, 0.097] | 0.049 [-0.005, 0.100] | 0.024 [0.002, 0.046] |
| dinov3_vs_dinov1 | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.078 [-0.117, -0.040] | -0.088 [-0.142, -0.032] | 0.031 [0.015, 0.048] |
| dinov3_vs_dinov1 | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.023 [-0.059, 0.013] | -0.019 [-0.059, 0.026] | 0.014 [0.000, 0.027] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | 375 | 0.027 [-0.043, 0.095] | 0.052 [-0.011, 0.117] | -0.025 [-0.058, 0.008] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.022 [-0.059, 0.100] | 0.037 [-0.052, 0.117] | 0.000 [-0.035, 0.037] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | 0.018 [-0.055, 0.089] | 0.036 [-0.032, 0.097] | -0.022 [-0.059, 0.016] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | -0.001 [-0.064, 0.064] | -0.005 [-0.075, 0.066] | 0.006 [-0.022, 0.035] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | 375 | 0.021 [-0.053, 0.092] | 0.048 [-0.019, 0.116] | -0.037 [-0.072, 0.001] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.040 [-0.038, 0.118] | 0.040 [-0.048, 0.122] | -0.012 [-0.048, 0.025] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | 0.027 [-0.044, 0.098] | 0.042 [-0.028, 0.105] | -0.027 [-0.063, 0.008] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | -0.000 [-0.062, 0.064] | 0.002 [-0.068, 0.075] | 0.003 [-0.025, 0.030] |
| dinov3_vs_dinov1 | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | 0.012 [-0.037, 0.061] | -0.014 [-0.064, 0.037] | 0.041 [0.012, 0.071] |
| dinov3_vs_dinov1 | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.013 [-0.035, 0.060] | 0.003 [-0.053, 0.061] | 0.035 [0.012, 0.058] |
| dinov3_vs_dinov1 | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.025 [-0.056, 0.004] | -0.015 [-0.059, 0.031] | 0.006 [-0.004, 0.016] |
| dinov3_vs_dinov1 | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.035 [-0.074, 0.003] | -0.037 [-0.085, 0.013] | 0.030 [0.014, 0.046] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | 375 | -0.011 [-0.092, 0.072] | -0.062 [-0.130, 0.008] | -0.041 [-0.087, 0.004] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.034 [-0.036, 0.103] | 0.013 [-0.057, 0.088] | 0.006 [-0.025, 0.037] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | 0.025 [-0.039, 0.094] | -0.042 [-0.104, 0.030] | -0.034 [-0.077, 0.009] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | 0.014 [-0.056, 0.086] | 0.005 [-0.082, 0.093] | 0.012 [-0.019, 0.044] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | 375 | -0.027 [-0.103, 0.052] | -0.072 [-0.136, -0.006] | -0.057 [-0.101, -0.013] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.035 [-0.035, 0.103] | 0.013 [-0.057, 0.088] | 0.005 [-0.025, 0.036] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | 0.034 [-0.033, 0.101] | -0.031 [-0.094, 0.044] | -0.041 [-0.083, 0.000] |
| dinov3_vs_dinov1 | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | 0.016 [-0.054, 0.087] | 0.007 [-0.080, 0.095] | 0.012 [-0.020, 0.044] |
| local_vs_global | full_808 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.003 [-0.051, 0.056] | 0.017 [-0.036, 0.068] | 0.041 [0.017, 0.064] |
| local_vs_global | full_808 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.011 [-0.037, 0.018] | -0.021 [-0.061, 0.023] | 0.008 [0.000, 0.016] |
| local_vs_global | radiomics_complete_case_375 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.014 [-0.062, 0.090] | -0.005 [-0.069, 0.064] | 0.022 [-0.004, 0.047] |
| local_vs_global | radiomics_complete_case_375 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | 0.010 [-0.051, 0.074] | -0.005 [-0.079, 0.061] | -0.012 [-0.044, 0.018] |
| local_vs_global | radiomics_complete_case_375 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.010 [-0.066, 0.086] | -0.005 [-0.071, 0.064] | 0.026 [-0.001, 0.053] |
| local_vs_global | radiomics_complete_case_375 | T0 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | 0.010 [-0.051, 0.075] | -0.003 [-0.076, 0.063] | -0.009 [-0.040, 0.021] |
| local_vs_global | full_808 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.104 [0.051, 0.156] | 0.085 [0.031, 0.137] | -0.045 [-0.074, -0.016] |
| local_vs_global | full_808 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | 0.023 [-0.021, 0.069] | 0.012 [-0.044, 0.070] | -0.015 [-0.035, 0.005] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.062 [-0.022, 0.143] | 0.046 [-0.036, 0.127] | -0.006 [-0.042, 0.031] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | 0.058 [-0.020, 0.134] | 0.029 [-0.043, 0.112] | -0.024 [-0.059, 0.012] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.074 [-0.010, 0.156] | 0.042 [-0.044, 0.132] | -0.019 [-0.057, 0.019] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | 0.043 [-0.034, 0.119] | 0.024 [-0.049, 0.107] | -0.017 [-0.051, 0.016] |
| local_vs_global | full_808 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.040 [-0.014, 0.092] | 0.038 [-0.011, 0.092] | -0.052 [-0.084, -0.020] |
| local_vs_global | full_808 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.045 [-0.092, 0.001] | -0.065 [-0.123, -0.007] | 0.026 [0.008, 0.045] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only_paired@LOCAL | 375 | 0.115 [0.031, 0.198] | 0.097 [0.034, 0.168] | -0.044 [-0.083, -0.004] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | 0.098 [0.015, 0.184] | 0.105 [0.021, 0.191] | -0.063 [-0.105, -0.023] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_ftv@LOCAL | 375 | 0.117 [0.031, 0.199] | 0.097 [0.033, 0.168] | -0.039 [-0.077, 0.001] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | 0.093 [0.008, 0.178] | 0.103 [0.016, 0.191] | -0.053 [-0.094, -0.013] |
| dinov3_vs_current_cnn_full | full_808 | T0 | GAP0_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | 0.033 [-0.023, 0.088] | 0.028 [-0.014, 0.074] | 0.009 [-0.005, 0.023] |
| dinov3_vs_current_cnn_full | full_808 | T0 | LOCAL0_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.021 [-0.033, 0.074] | 0.021 [-0.032, 0.071] | 0.020 [-0.004, 0.044] |
| dinov3_vs_current_cnn_full | full_808 | T0 | GAP0_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.017 [-0.047, 0.011] | -0.017 [-0.064, 0.032] | 0.003 [-0.005, 0.012] |
| dinov3_vs_current_cnn_full | full_808 | T0 | LOCAL0_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | 0.005 [-0.023, 0.033] | -0.001 [-0.049, 0.048] | 0.004 [-0.005, 0.012] |
| dinov3_vs_current_cnn_full | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | -0.012 [-0.071, 0.045] | -0.002 [-0.049, 0.046] | 0.067 [0.041, 0.093] |
| dinov3_vs_current_cnn_full | full_808 | T0-T1 | LOCAL0_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.045 [-0.009, 0.099] | 0.040 [-0.018, 0.100] | 0.023 [-0.001, 0.047] |
| dinov3_vs_current_cnn_full | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.081 [-0.123, -0.042] | -0.104 [-0.165, -0.046] | 0.042 [0.026, 0.060] |
| dinov3_vs_current_cnn_full | full_808 | T0-T1 | LOCAL0_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.027 [-0.064, 0.008] | -0.054 [-0.102, -0.001] | 0.017 [0.003, 0.031] |
| dinov3_vs_current_cnn_full | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@GLOBAL | 808 | -0.001 [-0.059, 0.055] | -0.011 [-0.065, 0.041] | 0.076 [0.046, 0.106] |
| dinov3_vs_current_cnn_full | full_808 | T0-T2 | LOCAL0_mri_only@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_only@LOCAL | 808 | 0.043 [-0.010, 0.094] | 0.049 [-0.001, 0.103] | 0.026 [0.002, 0.050] |
| dinov3_vs_current_cnn_full | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.015 [-0.053, 0.022] | -0.015 [-0.074, 0.041] | 0.012 [-0.001, 0.025] |
| dinov3_vs_current_cnn_full | full_808 | T0-T2 | LOCAL0_mri_clinical@LOCAL | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.038 [-0.082, 0.005] | -0.046 [-0.101, 0.010] | 0.027 [0.008, 0.047] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.006 [-0.035, 0.022] | -0.009 [-0.056, 0.037] | -0.002 [-0.010, 0.006] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.016 [-0.047, 0.013] | -0.030 [-0.083, 0.024] | 0.006 [-0.002, 0.014] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | -0.088 [-0.150, -0.026] | -0.099 [-0.179, -0.014] | 0.043 [0.015, 0.071] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | -0.078 [-0.136, -0.021] | -0.104 [-0.184, -0.028] | 0.031 [0.008, 0.054] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.076 [-0.118, -0.034] | -0.101 [-0.161, -0.037] | 0.036 [0.019, 0.054] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.053 [-0.093, -0.014] | -0.089 [-0.145, -0.029] | 0.022 [0.007, 0.037] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | -0.126 [-0.193, -0.060] | -0.121 [-0.207, -0.045] | 0.048 [0.018, 0.080] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | -0.068 [-0.138, 0.004] | -0.093 [-0.184, -0.002] | 0.024 [-0.002, 0.051] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@GLOBAL | 808 | -0.027 [-0.064, 0.008] | -0.031 [-0.087, 0.028] | 0.011 [-0.001, 0.023] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical@LOCAL | 808 | -0.072 [-0.116, -0.031] | -0.096 [-0.156, -0.033] | 0.037 [0.019, 0.056] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@GLOBAL | 375 | -0.150 [-0.222, -0.076] | -0.162 [-0.245, -0.077] | 0.086 [0.053, 0.119] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_paired@LOCAL | 375 | -0.052 [-0.128, 0.026] | -0.057 [-0.159, 0.048] | 0.023 [-0.008, 0.053] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | -0.079 [-0.142, -0.016] | -0.101 [-0.183, -0.016] | 0.043 [0.017, 0.070] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | -0.069 [-0.127, -0.010] | -0.103 [-0.186, -0.025] | 0.034 [0.012, 0.057] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | -0.146 [-0.213, -0.078] | -0.175 [-0.268, -0.089] | 0.061 [0.030, 0.092] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | -0.103 [-0.175, -0.029] | -0.151 [-0.247, -0.054] | 0.044 [0.015, 0.072] |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@GLOBAL | 375 | -0.226 [-0.300, -0.150] | -0.250 [-0.345, -0.159] | 0.118 [0.083, 0.154] |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dinov3_vitb16_lvd1689m_posthoc_mri_clinical_ftv@LOCAL | 375 | -0.133 [-0.208, -0.057] | -0.147 [-0.255, -0.043] | 0.065 [0.030, 0.100] |

## HR/HER2 binary phenotype probes：DINOv3 vs DINO v1 matched public aggregates

以下 probe 比较只给 pooled aggregate 的两组绝对值与 v3−v1 差值，不计算 CI，也不进入84-spec配对 bootstrap。

| 任务 | 空间 | n | AUROC DINOv1→v3 (Δ) | AUPRC DINOv1→v3 (Δ) | Brier DINOv1→v3 (Δ) |
| --- | --- | --- | --- | --- | --- |
| HR | GLOBAL | 808 | 0.522 → 0.505 (Δ -0.016) | 0.583 → 0.586 (Δ 0.004) | 0.283 → 0.351 (Δ 0.068) |
| HR | LOCAL | 808 | 0.596 → 0.577 (Δ -0.020) | 0.629 → 0.608 (Δ -0.021) | 0.274 → 0.285 (Δ 0.012) |
| HER2 | GLOBAL | 808 | 0.523 → 0.559 (Δ 0.036) | 0.263 → 0.294 (Δ 0.031) | 0.269 → 0.275 (Δ 0.006) |
| HER2 | LOCAL | 808 | 0.557 → 0.538 (Δ -0.019) | 0.299 → 0.275 (Δ -0.024) | 0.253 → 0.264 (Δ 0.010) |

## HR/HER2 四分类 subtype probe：DINOv3 vs DINO v1

| 空间 | n | macro AUROC DINOv1→v3 (Δ) | macro AUPRC DINOv1→v3 (Δ) | Brier DINOv1→v3 (Δ) | 准确率 DINOv1→v3 (Δ) |
| --- | --- | --- | --- | --- | --- |
| GLOBAL | 808 | 0.541 → 0.561 (Δ 0.020) | 0.282 → 0.286 (Δ 0.004) | 0.784 → 0.930 (Δ 0.147) | 0.345 → 0.314 (Δ -0.031) |
| LOCAL | 808 | 0.564 → 0.551 (Δ -0.013) | 0.301 → 0.285 (Δ -0.016) | 0.813 → 0.763 (Δ -0.050) | 0.339 → 0.332 (Δ -0.007) |

## FTV 与 ΔFTV decodability：DINOv3 vs DINO v1

Spearman/R² 的正差值表示 DINOv3 数值更高；RMSE/MAE 的负差值表示误差更低。这里仍是无 CI 的描述性 matched public aggregate 对照。

| 空间 | 任务 | 终点 | n | Spearman DINOv1→v3 (Δ) | R² DINOv1→v3 (Δ) | RMSE DINOv1→v3 (Δ) | MAE DINOv1→v3 (Δ) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLOBAL | static | T0 | 375 | 0.620 → 0.553 (Δ -0.067) | 0.395 → 0.335 (Δ -0.060) | 33.176 → 34.789 (Δ 1.613) | 15.996 → 17.108 (Δ 1.112) |
| GLOBAL | static | T1 | 375 | 0.574 → 0.510 (Δ -0.064) | 0.399 → 0.469 (Δ 0.071) | 29.215 → 27.445 (Δ -1.770) | 10.631 → 10.681 (Δ 0.050) |
| GLOBAL | static | T2 | 375 | 0.523 → 0.452 (Δ -0.071) | 0.326 → 0.204 (Δ -0.122) | 25.543 → 27.757 (Δ 2.214) | 6.915 → 7.529 (Δ 0.614) |
| GLOBAL | static | T3 | 375 | 0.347 → 0.299 (Δ -0.048) | 0.065 → 0.012 (Δ -0.053) | 11.997 → 12.334 (Δ 0.337) | 3.213 → 3.303 (Δ 0.091) |
| GLOBAL | delta | T0-T1 | 375 | 0.336 → 0.196 (Δ -0.139) | 0.117 → -0.011 (Δ -0.128) | 23.806 → 25.468 (Δ 1.662) | 15.001 → 15.490 (Δ 0.490) |
| GLOBAL | delta | T1-T2 | 375 | 0.297 → 0.194 (Δ -0.102) | 0.079 → -0.036 (Δ -0.115) | 16.497 → 17.498 (Δ 1.001) | 10.304 → 10.733 (Δ 0.429) |
| GLOBAL | delta | T2-T3 | 375 | 0.267 → 0.268 (Δ 0.000) | 0.083 → 0.014 (Δ -0.069) | 26.229 → 27.193 (Δ 0.964) | 12.215 → 13.358 (Δ 1.143) |
| LOCAL | static | T0 | 375 | 0.834 → 0.813 (Δ -0.021) | 0.623 → 0.522 (Δ -0.102) | 26.184 → 29.512 (Δ 3.328) | 10.549 → 12.144 (Δ 1.595) |
| LOCAL | static | T1 | 375 | 0.726 → 0.680 (Δ -0.046) | 0.323 → 0.268 (Δ -0.055) | 31.005 → 32.233 (Δ 1.229) | 9.618 → 10.364 (Δ 0.746) |
| LOCAL | static | T2 | 375 | 0.576 → 0.550 (Δ -0.027) | 0.155 → 0.176 (Δ 0.022) | 28.598 → 28.229 (Δ -0.369) | 6.955 → 7.046 (Δ 0.091) |
| LOCAL | static | T3 | 375 | 0.373 → 0.299 (Δ -0.074) | 0.113 → 0.063 (Δ -0.050) | 11.687 → 12.014 (Δ 0.327) | 3.169 → 3.266 (Δ 0.097) |
| LOCAL | delta | T0-T1 | 375 | 0.508 → 0.439 (Δ -0.068) | 0.225 → 0.110 (Δ -0.115) | 22.301 → 23.903 (Δ 1.602) | 12.451 → 13.501 (Δ 1.049) |
| LOCAL | delta | T1-T2 | 375 | 0.488 → 0.390 (Δ -0.097) | 0.225 → 0.123 (Δ -0.102) | 15.138 → 16.099 (Δ 0.961) | 8.847 → 9.092 (Δ 0.245) |
| LOCAL | delta | T2-T3 | 375 | 0.288 → 0.306 (Δ 0.018) | 0.152 → 0.178 (Δ 0.026) | 25.220 → 24.828 (Δ -0.391) | 11.725 → 11.197 (Δ -0.528) |

## 最终解释限制

这些表和公开图完整展示所有冻结候选/axes/timings；任何看似有利或不利的单格都不得替代全矩阵判断。由于这是结果公开后的单模型敏感性扩展，且 LVD-1689M 的 patient-level contamination 不能排除，本报告不声称证明 DINOv3 的一般优势，也不改变原 foundation-MRI 正式报告的结论。

<!-- FOUNDATION_MRI_DINOV3_GIT_HANDOFF_V1 -->
