# DCE-MRI Foundation Encoder Baselines：正式公开结果报告

本报告由冻结的 public-only finalizer 从公开 aggregate CSV/JSON 一次性生成。生成器不接受 private prediction、patient identifier、模型筛选、metric 排序或 top-k 参数；所有数值均按预注册 identity 字典序完整呈现。AUROC/AUPRC/Brier 配对差值统一为“候选减参照”。区间为固定患者级 bootstrap 的描述性 95% percentile interval，不作显著性检验、等效性判断或模型选择。

## 1. 全覆盖验收

| Artifact/cell family | Expected | Observed | Status |
| --- | --- | --- | --- |
| pCR pooled full | 39 | 39 | PASS |
| pCR pooled complete case | 87 | 87 | PASS |
| pCR pooled total | 126 | 126 | PASS |
| Paired comparisons | 132 | 132 | PASS |
| Paired metric rows | 396 | 396 | PASS |
| Phenotype pooled | 12 | 12 | PASS |
| Subtype pooled | 6 | 6 | PASS |
| FTV pooled | 42 | 42 | PASS |
| Baseline public pooled+macro rows | 252 | 252 | PASS |
| Phenotype public pooled+macro rows | 24 | 24 | PASS |
| Subtype public pooled+macro rows | 12 | 12 | PASS |
| FTV public pooled+macro rows | 84 | 84 | PASS |

正式 foundation 集合固定为 MedicalNet 3DSeg-8 ResNet-50 与 DINO v1 ViT-B/16。Current-CNN 配套参照固定为 `GAP0@GLOBAL` 与 `LOCAL0@LOCAL`。报告正文和附录均不得因结果方向删除任何 model、spatial axis、timing、population 或 endpoint。

## 2. 模型选择与完整 provenance

| Model | Pretraining/domain | Source revision | License | Parameters | Feature dim/visit | Checkpoint SHA-256 | Feature SHA-256 | Selection reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DINO v1 ViT-B/16 | ImageNet-1K self-supervised learning | Meta DINO native ViT-B/16 ImageNet-1K checkpoint | Apache-2.0 | 85798656 | 1536 | bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e | c078cd4ddc0c745c32ebcca247d44ef8025d08495f6e3193a481563f0d53ffbc | 公开 non-gated、训练语料固定的强通用 SSL reference |
| MedicalNet 3DSeg-8 ResNet-50 | 8 enumerated 3-D medical segmentation datasets (MRI+CT) | Tencent MedicalNet official archive, pretrain/resnet_50.pth | MIT | 46155072 | 14336 | 5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84 | ca45a46bd62e18e42b6d3f2426ce4690a4f3dbf7c2f44804ab0d19bd333ee4a2 | 可审计的 3-D medical reference，保留完整 DCE7 channel |

选择边界详见 [foundation model selection](foundation_model_selection.md)；完整执行状态见 [model execution ledger](model_execution_ledger.md)；current-CNN 来源见 [current CNN provenance audit](current_cnn_provenance_audit.md)。

## 3. 十二问

| # | 问题 | 固定规则答案 |
| --- | --- | --- |
| 1 | 使用了哪些 foundation models？ | MedicalNet 3DSeg-8 ResNet-50 与 DINO v1 ViT-B/16；完整 checkpoint/feature/source/license 见模型 provenance 表。 |
| 2 | 为什么选择它们？ | 分别提供可审计的 3-D medical reference 与固定 ImageNet-1K 的公开通用 SSL reference；选择理由、参数量与维度完整列出。 |
| 3 | MRI-only AUROC/AUPRC 是多少？ | 完整 808 人的 18 个 MRI-only cells（其中 foundation 12 个）全部列于 pCR 长表。 |
| 4 | LOCAL vs GLOBAL 如何？ | 证据混合，不能确认；预定 6 cells 中 favorable/mixed/adverse=1/3/2。 |
| 5 | Foundation vs current CNN LOCAL 如何？ | 证据混合，不能确认；预定 6 cells 中 favorable/mixed/adverse=1/3/2。 |
| 6 | HR/HER2 decodability 如何？ | 全部 12 个 HR/HER2 与 6 个 subtype pooled cells 均列出；不使用事后阈值筛选。 |
| 7 | FTV decodability 如何？ | 全部 42 个 FTV/ΔFTV pooled cells 均列出，其中 foundation 28 个。 |
| 8 | Clinical+Foundation 是否超过 Clinical-only？ | 证据混合，不能确认；预定 12 cells 中 favorable/mixed/adverse=0/2/10。 |
| 9 | Clinical+FTV+Foundation 是否超过 Clinical+FTV？ | 全部预定 cells 方向一致不利；预定 12 cells 中 favorable/mixed/adverse=0/0/12。 |
| 10 | 是否学到 tumor-size 以外信息？ | 未建立 FTV 以外增量 |
| 11 | 当前 World Model 是否明显 underuse MRI？ | 仅报告点估计方向：证据混合，不能确认；预定 12 cells 中 favorable/mixed/adverse=1/8/3。该描述不以 0 阈值证明‘明显’ underuse。 |
| 12 | 下一步是否值得替换/增强 encoder？ | 本实验不足以支持优先替换 encoder |

## 4. 固定规则 scientific diagnosis

固定 cell 规则为：AUROC 与 AUPRC 差值均大于 0 且 Brier 差值不大于 0 才记作 favorable；二者均小于 0 且 Brier 不小于 0 才记作 adverse；其余全部记作 mixed。只有预定集合中的每个 cell 均 favorable 才写“方向一致支持”。

- **Q4**：证据混合，不能确认；预定 6 cells 中 favorable/mixed/adverse=1/3/2。
  - AUROC descriptive CI favorable/inconclusive/adverse=0/6/0。
  - AUPRC descriptive CI favorable/inconclusive/adverse=0/6/0。
  - BRIER descriptive CI favorable/inconclusive/adverse=3/2/1。
- **Q5**：证据混合，不能确认；预定 6 cells 中 favorable/mixed/adverse=1/3/2。
  - AUROC descriptive CI favorable/inconclusive/adverse=0/6/0。
  - AUPRC descriptive CI favorable/inconclusive/adverse=0/6/0。
  - BRIER descriptive CI favorable/inconclusive/adverse=0/5/1。
- **Q8**：证据混合，不能确认；预定 12 cells 中 favorable/mixed/adverse=0/2/10。
  - AUROC descriptive CI favorable/inconclusive/adverse=0/8/4。
  - AUPRC descriptive CI favorable/inconclusive/adverse=0/6/6。
  - BRIER descriptive CI favorable/inconclusive/adverse=0/9/3。
- **Q9**：全部预定 cells 方向一致不利；预定 12 cells 中 favorable/mixed/adverse=0/0/12。
  - AUROC descriptive CI favorable/inconclusive/adverse=0/2/10。
  - AUPRC descriptive CI favorable/inconclusive/adverse=0/3/9。
  - BRIER descriptive CI favorable/inconclusive/adverse=0/2/10。
- **Q11**：证据混合，不能确认；预定 12 cells 中 favorable/mixed/adverse=1/8/3。
  - AUROC descriptive CI favorable/inconclusive/adverse=0/12/0。
  - AUPRC descriptive CI favorable/inconclusive/adverse=1/11/0。
  - BRIER descriptive CI favorable/inconclusive/adverse=0/5/7。
- **Q10**：未建立 FTV 以外增量。
- **Q12**：本实验不足以支持优先替换 encoder。
- 所有区间均为描述性，不作显著性检验、等效性判断或 test-driven selection。

## 5. 全部 pCR pooled OOF cells

本表完整保留 808 人 primary 与 375 人 complete-case sensitivity。`FTV representation / readout` 的左侧说明 representation 是否接受 FTV supervision，右侧说明 pCR readout 是否把 FTV 作为 covariate；两者不得混写。

| Population | Model | Pretraining | Spatial | MRI | Clinical | Timing | FTV representation / readout | n | pCR+ | AUROC | AUPRC | Brier | Calibration intercept | Calibration slope | ECE10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_808 | GAP0_mri_clinical | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.720198 | 0.565856 | 0.218278 | -0.717525 | 1.236176 | 0.150856 |
| full_808 | GAP0_mri_clinical | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.714972 | 0.560802 | 0.217669 | -0.692266 | 0.974991 | 0.147186 |
| full_808 | GAP0_mri_clinical | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.696490 | 0.541466 | 0.222712 | -0.641058 | 0.663856 | 0.145287 |
| full_808 | GAP0_mri_only | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.494897 | 0.333301 | 0.259364 | -0.667755 | -0.109274 | 0.167572 |
| full_808 | GAP0_mri_only | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.513655 | 0.352282 | 0.266245 | 2.033991 | 2.272428 | 0.178877 |
| full_808 | GAP0_mri_only | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.552025 | 0.394256 | 0.274200 | 3.065532 | 2.923969 | 0.195024 |
| full_808 | LOCAL0_mri_clinical | current response-state (grounded=false) | LOCAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.687962 | 0.528620 | 0.225467 | -0.669236 | 0.803243 | 0.142569 |
| full_808 | LOCAL0_mri_clinical | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.683896 | 0.522358 | 0.228395 | -0.668129 | 0.729352 | 0.149869 |
| full_808 | LOCAL0_mri_clinical | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.674658 | 0.507697 | 0.233342 | -0.621132 | 0.409976 | 0.153897 |
| full_808 | LOCAL0_mri_only | current response-state (grounded=false) | LOCAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.509691 | 0.357967 | 0.288992 | -0.657727 | 0.040837 | 0.213335 |
| full_808 | LOCAL0_mri_only | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.560123 | 0.395518 | 0.264620 | -0.647012 | 0.136320 | 0.181301 |
| full_808 | LOCAL0_mri_only | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.546983 | 0.371648 | 0.272114 | -0.642479 | 0.094086 | 0.186310 |
| full_808 | clinical_only | 不适用 | NONE | 否 | 是 | T0 | 不适用 / 否 | 808 | 275 | 0.709156 | 0.557514 | 0.223800 | -0.681309 | 1.071307 | 0.147748 |
| full_808 | clinical_only | 不适用 | NONE | 否 | 是 | T0-T1 | 不适用 / 否 | 808 | 275 | 0.709156 | 0.557514 | 0.223800 | -0.681309 | 1.071307 | 0.147748 |
| full_808 | clinical_only | 不适用 | NONE | 否 | 是 | T0-T2 | 不适用 / 否 | 808 | 275 | 0.709156 | 0.557514 | 0.223800 | -0.681309 | 1.071307 | 0.147748 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.702801 | 0.529114 | 0.227596 | -0.718922 | 1.439649 | 0.159783 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.711083 | 0.544139 | 0.228475 | -0.764071 | 2.197061 | 0.160021 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.707010 | 0.541912 | 0.228758 | -0.750482 | 2.157327 | 0.160418 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.668572 | 0.498009 | 0.236861 | -0.629575 | 0.413755 | 0.163751 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.679816 | 0.487824 | 0.231618 | -0.659156 | 0.701528 | 0.148495 |
| full_808 | dino_vitb16_imagenet1k_mri_clinical | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.672161 | 0.498463 | 0.231215 | -0.637201 | 0.641724 | 0.149977 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.535610 | 0.380071 | 0.280043 | 2.917635 | 2.544128 | 0.205954 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.550230 | 0.390247 | 0.288494 | 3.045243 | 2.214718 | 0.212646 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.539110 | 0.396833 | 0.308978 | -4.694977 | 1.000000 | 0.265120 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.527580 | 0.373463 | 0.303900 | 4.375979 | 1.457271 | 0.236988 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.557025 | 0.386558 | 0.263821 | -0.645575 | 0.171986 | 0.175967 |
| full_808 | dino_vitb16_imagenet1k_mri_only | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.577588 | 0.417573 | 0.263012 | -0.666329 | 0.261554 | 0.188856 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.663152 | 0.481765 | 0.248173 | 2.086338 | 1.000000 | 0.183664 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.706495 | 0.518737 | 0.229486 | -0.759138 | 2.125603 | 0.160385 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.674119 | 0.488471 | 0.233199 | -0.655625 | 0.340488 | 0.153512 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0 | 否 / 否 | 808 | 275 | 0.703142 | 0.508576 | 0.231388 | -0.680452 | 0.588357 | 0.165283 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 808 | 275 | 0.710435 | 0.523867 | 0.229157 | -0.751006 | 2.028419 | 0.160378 |
| full_808 | medicalnet_resnet50_3dseg8_mri_clinical | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 808 | 275 | 0.703394 | 0.518261 | 0.229640 | -0.740047 | 1.912662 | 0.159658 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.538663 | 0.378409 | 0.301796 | 1.385837 | 1.000000 | 0.238573 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.536660 | 0.375549 | 0.286683 | 2.469585 | 1.886850 | 0.212902 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.521221 | 0.368101 | 0.315389 | 1.514952 | 1.000000 | 0.254190 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0 | 否 / 否 | 808 | 275 | 0.527020 | 0.366359 | 0.291650 | 4.102069 | 1.000000 | 0.216694 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 808 | 275 | 0.507992 | 0.348456 | 0.295484 | 4.548102 | 1.000000 | 0.237454 |
| full_808 | medicalnet_resnet50_3dseg8_mri_only | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 808 | 275 | 0.526986 | 0.353583 | 0.288481 | 4.216461 | 1.000000 | 0.216214 |
| radiomics_complete_case_375 | GAP0_mri_clinical_ftv | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.663945 | 0.478780 | 0.230502 | -0.853508 | 0.529963 | 0.187422 |
| radiomics_complete_case_375 | GAP0_mri_clinical_ftv | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.689846 | 0.484316 | 0.228454 | -0.876800 | 0.514151 | 0.196645 |
| radiomics_complete_case_375 | GAP0_mri_clinical_ftv | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.696638 | 0.508507 | 0.222150 | 0.271593 | 3.080142 | 0.173607 |
| radiomics_complete_case_375 | GAP0_mri_clinical_paired | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.655437 | 0.464478 | 0.231624 | -0.839000 | 0.472241 | 0.186264 |
| radiomics_complete_case_375 | GAP0_mri_clinical_paired | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.650875 | 0.464705 | 0.241323 | -0.826373 | 0.327221 | 0.212741 |
| radiomics_complete_case_375 | GAP0_mri_clinical_paired | current response-state (grounded=false) | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.643259 | 0.457890 | 0.248207 | 0.614981 | 3.464898 | 0.214929 |
| radiomics_complete_case_375 | GAP0_mri_ftv | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.485352 | 0.282589 | 0.261263 | -0.870014 | 0.045875 | 0.193843 |
| radiomics_complete_case_375 | GAP0_mri_ftv | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.495883 | 0.296863 | 0.305196 | -1.257715 | 3.371910 | 0.261101 |
| radiomics_complete_case_375 | GAP0_mri_ftv | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.567033 | 0.333308 | 0.289277 | -4.685704 | 2.377820 | 0.242880 |
| radiomics_complete_case_375 | GAP0_mri_only_paired | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.482676 | 0.281189 | 0.261828 | -0.875993 | 0.014909 | 0.206041 |
| radiomics_complete_case_375 | GAP0_mri_only_paired | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.455506 | 0.281397 | 0.321398 | -2.208289 | 2.673960 | 0.280631 |
| radiomics_complete_case_375 | GAP0_mri_only_paired | current response-state (grounded=false) | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.496810 | 0.291591 | 0.307228 | -3.009851 | 1.386515 | 0.269569 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_ftv | current response-state (grounded=false) | LOCAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.670292 | 0.463160 | 0.237061 | -0.990750 | 1.152574 | 0.213465 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_ftv | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.643602 | 0.427098 | 0.250863 | 4.632788 | 2.428898 | 0.191303 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_ftv | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.634099 | 0.429497 | 0.297026 | -2.423272 | 1.000000 | 0.279298 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_paired | current response-state (grounded=false) | LOCAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.670257 | 0.463235 | 0.237026 | -0.990488 | 1.152779 | 0.213404 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_paired | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.638937 | 0.443210 | 0.247399 | 0.978881 | 3.363336 | 0.210894 |
| radiomics_complete_case_375 | LOCAL0_mri_clinical_paired | current response-state (grounded=false) | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.606861 | 0.400852 | 0.277726 | -4.831996 | 2.216475 | 0.240718 |
| radiomics_complete_case_375 | LOCAL0_mri_ftv | current response-state (grounded=false) | LOCAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.497770 | 0.303180 | 0.296408 | 3.138215 | 2.711685 | 0.254289 |
| radiomics_complete_case_375 | LOCAL0_mri_ftv | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.519160 | 0.323745 | 0.277152 | 1.938562 | 1.293165 | 0.236624 |
| radiomics_complete_case_375 | LOCAL0_mri_ftv | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.544597 | 0.311487 | 0.351394 | -0.879249 | 1.000000 | 0.328542 |
| radiomics_complete_case_375 | LOCAL0_mri_only_paired | current response-state (grounded=false) | LOCAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.502436 | 0.304451 | 0.296030 | 2.996779 | 2.697693 | 0.257197 |
| radiomics_complete_case_375 | LOCAL0_mri_only_paired | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.497101 | 0.312053 | 0.271379 | -0.882177 | -0.069258 | 0.235080 |
| radiomics_complete_case_375 | LOCAL0_mri_only_paired | current response-state (grounded=false) | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.530051 | 0.316806 | 0.337453 | -3.747770 | 1.497830 | 0.312802 |
| radiomics_complete_case_375 | clinical_ftv | 不适用 | TABULAR | 否 | 是 | T0 | 不适用 / 是 | 375 | 110 | 0.679640 | 0.484537 | 0.234203 | -0.899699 | 0.674136 | 0.198843 |
| radiomics_complete_case_375 | clinical_ftv | 不适用 | TABULAR | 否 | 是 | T0-T1 | 不适用 / 是 | 375 | 110 | 0.724734 | 0.550675 | 0.215025 | -0.851958 | 0.523791 | 0.181736 |
| radiomics_complete_case_375 | clinical_ftv | 不适用 | TABULAR | 否 | 是 | T0-T2 | 不适用 / 是 | 375 | 110 | 0.772967 | 0.578171 | 0.194945 | -0.805141 | 0.527552 | 0.153332 |
| radiomics_complete_case_375 | clinical_only_paired | 不适用 | NONE | 否 | 是 | T0 | 不适用 / 否 | 375 | 110 | 0.689057 | 0.485515 | 0.237517 | -0.898416 | 0.577243 | 0.208864 |
| radiomics_complete_case_375 | clinical_only_paired | 不适用 | NONE | 否 | 是 | T0-T1 | 不适用 / 否 | 375 | 110 | 0.689057 | 0.485515 | 0.237517 | -0.898416 | 0.577243 | 0.208864 |
| radiomics_complete_case_375 | clinical_only_paired | 不适用 | NONE | 否 | 是 | T0-T2 | 不适用 / 否 | 375 | 110 | 0.689057 | 0.485515 | 0.237517 | -0.898416 | 0.577243 | 0.208864 |
| radiomics_complete_case_375 | clinical_radiomics | 不适用 | TABULAR | 否 | 是 | T0 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.721269 | 0.507929 | 0.222027 | -0.878387 | 1.116992 | 0.194891 |
| radiomics_complete_case_375 | clinical_radiomics | 不适用 | TABULAR | 否 | 是 | T0-T1 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.731835 | 0.539953 | 0.213547 | -0.827477 | 0.707097 | 0.175555 |
| radiomics_complete_case_375 | clinical_radiomics | 不适用 | TABULAR | 否 | 是 | T0-T2 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.756278 | 0.571215 | 0.203494 | -0.843579 | 0.680694 | 0.165088 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.587822 | 0.396084 | 0.268359 | -4.783176 | 1.310852 | 0.237209 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.551630 | 0.333267 | 0.303036 | -4.763554 | 1.063210 | 0.260401 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.513242 | 0.359427 | 0.354283 | -3.092678 | 1.000000 | 0.354647 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.605455 | 0.387914 | 0.260924 | -3.791649 | 2.915045 | 0.229914 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.621955 | 0.397629 | 0.256086 | -2.731183 | 3.253443 | 0.204993 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.623259 | 0.424385 | 0.248446 | 1.214886 | 3.443263 | 0.207668 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.587273 | 0.395833 | 0.269706 | -4.629816 | 1.000000 | 0.241952 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.545146 | 0.327650 | 0.307394 | 2.441005 | 1.000000 | 0.266851 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.514031 | 0.365701 | 0.357271 | -2.694240 | 1.000000 | 0.359554 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.605111 | 0.387930 | 0.260939 | -3.794736 | 2.910141 | 0.229878 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.622264 | 0.397771 | 0.255651 | -2.738980 | 3.248374 | 0.194534 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_clinical_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.622916 | 0.424068 | 0.248414 | 1.204291 | 3.442710 | 0.207572 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.515712 | 0.316895 | 0.273966 | 1.986404 | 1.080864 | 0.231450 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.511698 | 0.304030 | 0.327623 | -4.767264 | 1.000000 | 0.277867 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.511321 | 0.359078 | 0.368993 | -1.402061 | 1.000000 | 0.359850 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.540274 | 0.331862 | 0.271868 | -3.451529 | 3.546716 | 0.221016 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.566312 | 0.354092 | 0.283358 | -1.427480 | 2.824361 | 0.247223 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_ftv | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.566278 | 0.371206 | 0.268775 | 1.337133 | 3.195950 | 0.218801 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.515952 | 0.317101 | 0.273932 | 1.965543 | 1.094272 | 0.231425 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.499039 | 0.292916 | 0.314414 | -2.969949 | 1.000000 | 0.270147 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.495334 | 0.348368 | 0.359720 | -3.096016 | 1.000000 | 0.335470 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.540172 | 0.331752 | 0.271880 | -3.448178 | 3.546315 | 0.221036 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.566449 | 0.354112 | 0.283323 | -1.551152 | 2.825605 | 0.247052 |
| radiomics_complete_case_375 | dino_vitb16_imagenet1k_mri_only_paired | DINO v1 ImageNet-1K SSL | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.565111 | 0.369817 | 0.269211 | 1.363831 | 3.165096 | 0.214927 |
| radiomics_complete_case_375 | ftv_only | 不适用 | TABULAR | 否 | 否 | T0 | 不适用 / 是 | 375 | 110 | 0.477050 | 0.284264 | 0.250066 | -0.873356 | -17.569160 | 0.206762 |
| radiomics_complete_case_375 | ftv_only | 不适用 | TABULAR | 否 | 否 | T0-T1 | 不适用 / 是 | 375 | 110 | 0.661990 | 0.417299 | 0.242555 | -0.882212 | 0.775777 | 0.202293 |
| radiomics_complete_case_375 | ftv_only | 不适用 | TABULAR | 否 | 否 | T0-T2 | 不适用 / 是 | 375 | 110 | 0.687067 | 0.436515 | 0.236992 | -0.874954 | 0.544535 | 0.202175 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.621664 | 0.412322 | 0.244327 | -0.876722 | 0.309309 | 0.201813 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.554202 | 0.375608 | 0.259436 | 2.488225 | 1.728528 | 0.213163 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.571870 | 0.408935 | 0.252175 | 1.761551 | 2.943492 | 0.210349 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0 | 否 / 是 | 375 | 110 | 0.647925 | 0.424930 | 0.245158 | 2.289340 | 3.212120 | 0.218840 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T1 | 否 / 是 | 375 | 110 | 0.579640 | 0.366677 | 0.262383 | 4.371626 | 1.000000 | 0.232944 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T2 | 否 / 是 | 375 | 110 | 0.569005 | 0.351167 | 0.281372 | 4.643187 | 1.000000 | 0.242522 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.621321 | 0.408716 | 0.244923 | -0.877667 | 0.311232 | 0.200953 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.549091 | 0.373105 | 0.259675 | 1.248569 | 2.792028 | 0.215560 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.579863 | 0.418532 | 0.249181 | 1.661703 | 2.972162 | 0.215848 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0 | 否 / 否 | 375 | 110 | 0.647513 | 0.424665 | 0.245319 | 2.242464 | 3.176468 | 0.218994 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T1 | 否 / 否 | 375 | 110 | 0.544648 | 0.338225 | 0.272934 | 4.183940 | 1.000000 | 0.239009 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_clinical_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 是 | T0-T2 | 否 / 否 | 375 | 110 | 0.551544 | 0.339575 | 0.288332 | 4.596932 | 1.000000 | 0.254014 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.500515 | 0.308071 | 0.301733 | 1.728394 | 1.000000 | 0.255283 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.522950 | 0.339383 | 0.293511 | 2.507580 | 1.256195 | 0.266032 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.510943 | 0.336251 | 0.263619 | 1.342050 | 2.710098 | 0.218516 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0 | 否 / 是 | 375 | 110 | 0.498456 | 0.316331 | 0.281702 | 2.628182 | 1.447456 | 0.250741 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T1 | 否 / 是 | 375 | 110 | 0.500823 | 0.305658 | 0.298509 | 2.479303 | 1.000000 | 0.272385 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_ftv | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T2 | 否 / 是 | 375 | 110 | 0.532247 | 0.322758 | 0.301919 | -2.609167 | 1.000000 | 0.262412 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.514374 | 0.321664 | 0.287507 | 2.048114 | 3.100647 | 0.250617 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.549743 | 0.355071 | 0.278862 | 1.303039 | 3.206091 | 0.239640 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | GLOBAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.497050 | 0.322713 | 0.268220 | 2.034429 | 2.079698 | 0.227952 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0 | 否 / 否 | 375 | 110 | 0.498388 | 0.315895 | 0.281727 | 2.582342 | 1.423932 | 0.250751 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T1 | 否 / 否 | 375 | 110 | 0.510566 | 0.315148 | 0.287491 | 3.865865 | 1.000000 | 0.261456 |
| radiomics_complete_case_375 | medicalnet_resnet50_3dseg8_mri_only_paired | MedicalNet 3DSeg-8 | LOCAL | 是 | 否 | T0-T2 | 否 / 否 | 375 | 110 | 0.529640 | 0.323543 | 0.298065 | 1.776416 | 1.000000 | 0.260505 |
| radiomics_complete_case_375 | radiomics_only | 不适用 | TABULAR | 否 | 否 | T0 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.563774 | 0.349728 | 0.246695 | -0.882233 | 0.986843 | 0.204284 |
| radiomics_complete_case_375 | radiomics_only | 不适用 | TABULAR | 否 | 否 | T0-T1 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.687513 | 0.443409 | 0.228157 | -0.863264 | 0.804856 | 0.188292 |
| radiomics_complete_case_375 | radiomics_only | 不适用 | TABULAR | 否 | 否 | T0-T2 | 不适用 / radiomics bundle（含 FTV） | 375 | 110 | 0.754648 | 0.514597 | 0.211714 | -0.920965 | 0.972491 | 0.180552 |

## 6. 全部预注册 paired pCR comparisons

每个 comparison 恰好同时展示 ΔAUROC、ΔAUPRC 与 ΔBrier；绝对 pCR 分数没有 bootstrap interval，不得把本表的 paired interval 移贴到绝对分数。

| Comparison ID | Family | Estimand | Population | Timing | Reference | Candidate | n | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔBrier [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beyond_ftv:211bc51d6581 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.091818 [-0.155160, -0.029029] | -0.088452 [-0.176160, -0.002401] | 0.034156 [0.008441, 0.060863] |
| beyond_ftv:b5f177f7d0ef | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.074185 [-0.140999, -0.003698] | -0.096622 [-0.186590, 0.003594] | 0.026721 [0.002541, 0.051040] |
| beyond_ftv:6ccbdddb9d6c | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.057976 [-0.120318, 0.007253] | -0.072215 [-0.168783, 0.019226] | 0.010124 [-0.009911, 0.029773] |
| beyond_ftv:91ab093ef93f | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.031715 [-0.091002, 0.029292] | -0.059607 [-0.142735, 0.028366] | 0.010954 [-0.006275, 0.028364] |
| beyond_ftv:196febb128cd | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.173105 [-0.245987, -0.102575] | -0.217408 [-0.304678, -0.132345] | 0.088012 [0.053864, 0.122617] |
| beyond_ftv:a35e8ae4e163 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.102779 [-0.171406, -0.033528] | -0.153046 [-0.239898, -0.065712] | 0.041062 [0.013548, 0.070275] |
| beyond_ftv:de876b68af6d | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.170532 [-0.247362, -0.093224] | -0.175067 [-0.273144, -0.074353] | 0.044411 [0.018858, 0.070323] |
| beyond_ftv:941ac2a8ec5f | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.145094 [-0.221682, -0.066185] | -0.183998 [-0.279931, -0.083518] | 0.047358 [0.019930, 0.074836] |
| beyond_ftv:0eb3b35803c0 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.259726 [-0.338337, -0.182516] | -0.218744 [-0.321345, -0.129160] | 0.159338 [0.116475, 0.203329] |
| beyond_ftv:c0ab4f365ff9 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.149708 [-0.224509, -0.076406] | -0.153785 [-0.255262, -0.054445] | 0.053501 [0.022869, 0.084135] |
| beyond_ftv:d7648cd5ecb1 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.201098 [-0.272810, -0.128676] | -0.169236 [-0.271661, -0.071342] | 0.057229 [0.029110, 0.085229] |
| beyond_ftv:0dd3f92286b1 | beyond_ftv | clinical_FTV_foundation_minus_clinical_FTV | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.203962 [-0.280692, -0.127068] | -0.227004 [-0.326053, -0.132462] | 0.086426 [0.050523, 0.121618] |
| clinical_gain:b4bb38bc33db | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | 0.011042 [-0.013497, 0.035657] | 0.008342 [-0.041420, 0.057320] | -0.005522 [-0.013112, 0.002389] |
| clinical_gain:fb1e2a2a0e01 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.021194 [-0.050235, 0.007538] | -0.028894 [-0.081155, 0.024180] | 0.001667 [-0.007377, 0.010747] |
| clinical_gain:c75bab6e0258 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.006355 [-0.032787, 0.019561] | -0.028400 [-0.071561, 0.015469] | 0.003796 [-0.002621, 0.010175] |
| clinical_gain:5c1db8347fb0 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.040583 [-0.074519, -0.008869] | -0.059505 [-0.112564, -0.009353] | 0.013061 [0.002317, 0.024665] |
| clinical_gain:b11d88085537 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.046004 [-0.084577, -0.009931] | -0.075750 [-0.131922, -0.015999] | 0.024373 [0.009865, 0.040156] |
| clinical_gain:7c525284cfd2 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | -0.006014 [-0.035716, 0.024037] | -0.048938 [-0.093902, -0.000621] | 0.007589 [0.000375, 0.015386] |
| clinical_gain:de58d508b60e | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | 0.005816 [-0.016676, 0.028892] | 0.003288 [-0.041854, 0.049057] | -0.006130 [-0.013657, 0.001424] |
| clinical_gain:6a828e900674 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.025260 [-0.054808, 0.004902] | -0.035157 [-0.080921, 0.013059] | 0.004596 [-0.003768, 0.013028] |
| clinical_gain:5714e6470833 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | 0.001927 [-0.025753, 0.028660] | -0.013375 [-0.059439, 0.032810] | 0.004675 [-0.002135, 0.011355] |
| clinical_gain:6f75582945cb | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.029340 [-0.062687, 0.004118] | -0.069690 [-0.123540, -0.019142] | 0.007818 [-0.001923, 0.017961] |
| clinical_gain:a41de5496e43 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.002661 [-0.028496, 0.023064] | -0.038777 [-0.082532, 0.009264] | 0.005686 [-0.000836, 0.012284] |
| clinical_gain:3169367c3d31 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T1 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.001279 [-0.025540, 0.027395] | -0.033647 [-0.079808, 0.018693] | 0.005358 [-0.001431, 0.012325] |
| clinical_gain:f49644ff3702 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | -0.012666 [-0.042431, 0.018358] | -0.016048 [-0.060554, 0.034665] | -0.001088 [-0.011042, 0.008530] |
| clinical_gain:cb614e7102ca | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.034498 [-0.068695, -0.000628] | -0.049817 [-0.102202, 0.002304] | 0.009542 [-0.002537, 0.022409] |
| clinical_gain:30ab59072156 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.002146 [-0.031134, 0.026099] | -0.015602 [-0.063983, 0.031886] | 0.004959 [-0.001994, 0.011895] |
| clinical_gain:0b2f11870cff | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.036995 [-0.072770, -0.003263] | -0.059051 [-0.115421, -0.004731] | 0.007415 [-0.002738, 0.018178] |
| clinical_gain:d33dfe3067b2 | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.035037 [-0.070616, -0.001073] | -0.069043 [-0.125567, -0.007867] | 0.009400 [-0.002035, 0.021324] |
| clinical_gain:cc941b657d2d | clinical_gain | clinical_plus_MRI_minus_clinical | full_808 | T0-T2 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | -0.005762 [-0.032569, 0.020675] | -0.039253 [-0.086501, 0.012231] | 0.005840 [-0.001040, 0.013014] |
| clinical_gain:d696b4edba87 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.033619 [-0.080072, 0.012765] | -0.021037 [-0.083267, 0.036266] | -0.005893 [-0.021700, 0.010514] |
| clinical_gain:b1abf2471804 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.018799 [-0.058178, 0.021269] | -0.022280 [-0.075030, 0.030544] | -0.000491 [-0.011114, 0.009714] |
| clinical_gain:5acd57cd9797 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.101784 [-0.164546, -0.039153] | -0.089682 [-0.178200, -0.005054] | 0.032189 [0.006074, 0.059019] |
| clinical_gain:25f642eaeb52 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.083945 [-0.150844, -0.014368] | -0.097585 [-0.188056, -0.002052] | 0.023422 [-0.000823, 0.048351] |
| clinical_gain:2b74b8a243ad | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.067736 [-0.129949, -0.003275] | -0.076799 [-0.170440, 0.013263] | 0.007406 [-0.012993, 0.027539] |
| clinical_gain:aae31817bb71 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.041544 [-0.099798, 0.018051] | -0.060850 [-0.141889, 0.024126] | 0.007802 [-0.009799, 0.025969] |
| clinical_gain:23f97c7aff3e | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.038182 [-0.091982, 0.013932] | -0.020810 [-0.088158, 0.045148] | 0.003806 [-0.015987, 0.025269] |
| clinical_gain:c0c410eac217 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.050120 [-0.111395, 0.009299] | -0.042305 [-0.126911, 0.037231] | 0.009882 [-0.011123, 0.031424] |
| clinical_gain:07e110ab7fbc | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.143911 [-0.217819, -0.072992] | -0.157865 [-0.241043, -0.080347] | 0.069877 [0.038146, 0.102385] |
| clinical_gain:016fafd94e00 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.066792 [-0.134022, 0.001996] | -0.087744 [-0.176364, -0.000541] | 0.018134 [-0.008622, 0.045672] |
| clinical_gain:70d046ba3dbb | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.139966 [-0.216175, -0.063097] | -0.112411 [-0.204607, -0.016449] | 0.022158 [-0.000531, 0.045578] |
| clinical_gain:b89b0cf80d18 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.144408 [-0.223287, -0.064897] | -0.147291 [-0.238140, -0.056052] | 0.035417 [0.010908, 0.060394] |
| clinical_gain:3cb73731e52e | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.045798 [-0.110205, 0.014548] | -0.027625 [-0.108456, 0.046707] | 0.010690 [-0.014436, 0.036860] |
| clinical_gain:0627b9aaafcf | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.082196 [-0.153075, -0.010142] | -0.084663 [-0.179203, 0.010013] | 0.040209 [0.010354, 0.071473] |
| clinical_gain:03327f1708cf | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.175026 [-0.253622, -0.096975] | -0.119814 [-0.214331, -0.039143] | 0.119755 [0.079730, 0.161352] |
| clinical_gain:20c53598e7f9 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.066141 [-0.141509, 0.010978] | -0.061448 [-0.154491, 0.027124] | 0.010897 [-0.014358, 0.036969] |
| clinical_gain:c999a9515956 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.109194 [-0.185267, -0.034178] | -0.066983 [-0.162728, 0.032278] | 0.011664 [-0.012801, 0.037443] |
| clinical_gain:efaa4d0806a7 | clinical_gain | clinical_plus_MRI_minus_clinical | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.137513 [-0.212224, -0.060129] | -0.145940 [-0.236519, -0.054694] | 0.050815 [0.020303, 0.080541] |
| foundation_vs_current_cnn:528ddd2d991a | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.017397 [-0.045900, 0.011585] | -0.036742 [-0.090553, 0.017953] | 0.009318 [0.001580, 0.016956] |
| foundation_vs_current_cnn:9107d74ffd94 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.019389 [-0.053241, 0.012468] | -0.030611 [-0.083431, 0.020938] | 0.011394 [0.000378, 0.023579] |
| foundation_vs_current_cnn:c6608dbb3ec3 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | 0.040713 [-0.014469, 0.096738] | 0.046770 [0.000578, 0.092349] | 0.020680 [0.004679, 0.037673] |
| foundation_vs_current_cnn:51df8154a41b | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.017888 [-0.038076, 0.070949] | 0.015496 [-0.036855, 0.067099] | 0.014908 [-0.009087, 0.039861] |
| foundation_vs_current_cnn:062a5859f72d | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.057046 [-0.095234, -0.020484] | -0.084092 [-0.138603, -0.029173] | 0.029895 [0.015958, 0.045019] |
| foundation_vs_current_cnn:616bf2496481 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.015180 [-0.016581, 0.047893] | -0.020044 [-0.072122, 0.036835] | 0.005921 [-0.003617, 0.015459] |
| foundation_vs_current_cnn:482389bb9772 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | 0.043766 [-0.014350, 0.102048] | 0.045107 [-0.000200, 0.095049] | 0.042432 [0.022029, 0.062767] |
| foundation_vs_current_cnn:392dbdfc3ea8 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | 0.017329 [-0.036505, 0.069277] | 0.008392 [-0.040067, 0.054214] | 0.002658 [-0.018341, 0.024677] |
| foundation_vs_current_cnn:721e43c89aa6 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.003889 [-0.028072, 0.020273] | -0.016663 [-0.059811, 0.026950] | 0.010806 [0.003144, 0.018285] |
| foundation_vs_current_cnn:ade9f84356d3 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T1 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.004080 [-0.035874, 0.026857] | -0.034534 [-0.082934, 0.009959] | 0.003223 [-0.006753, 0.013763] |
| foundation_vs_current_cnn:e8481d3cf8af | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | 0.036575 [-0.020328, 0.094032] | 0.037965 [-0.013542, 0.092075] | 0.022249 [0.000565, 0.044237] |
| foundation_vs_current_cnn:96d1ca566c89 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T1 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | -0.003097 [-0.057967, 0.048701] | -0.008960 [-0.062878, 0.044555] | -0.000800 [-0.016824, 0.015986] |
| foundation_vs_current_cnn:7a429306e903 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.008477 [-0.031549, 0.014488] | -0.042065 [-0.084870, 0.000716] | 0.011816 [0.004434, 0.019337] |
| foundation_vs_current_cnn:96d6f761ec9c | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T1 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.026539 [-0.002165, 0.056074] | 0.001509 [-0.044184, 0.051808] | 0.000762 [-0.008421, 0.009730] |
| foundation_vs_current_cnn:039c5fd08755 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | 0.023005 [-0.029960, 0.077484] | 0.023266 [-0.025050, 0.074413] | 0.020438 [0.001326, 0.040233] |
| foundation_vs_current_cnn:f3afc56aa841 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T1 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.052130 [-0.106177, 0.000873] | -0.047062 [-0.092979, 0.000521] | 0.030864 [0.010427, 0.051506] |
| foundation_vs_current_cnn:a99f4e754348 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | 0.010520 [-0.019786, 0.040983] | 0.000445 [-0.049712, 0.049840] | 0.006046 [-0.003986, 0.015816] |
| foundation_vs_current_cnn:91e4a56d2dcd | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T2 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.002497 [-0.039875, 0.035416] | -0.009234 [-0.063319, 0.046042] | -0.002127 [-0.016197, 0.011885] |
| foundation_vs_current_cnn:cc1fa69535ca | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | -0.012915 [-0.066831, 0.042431] | 0.002577 [-0.054624, 0.059751] | 0.034778 [0.009872, 0.060433] |
| foundation_vs_current_cnn:986fc1f299cf | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T2 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.030605 [-0.022827, 0.081992] | 0.045926 [-0.005934, 0.098401] | -0.009102 [-0.026177, 0.008919] |
| foundation_vs_current_cnn:252e340a6a7e | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.022371 [-0.056838, 0.011364] | -0.052995 [-0.110259, 0.002325] | 0.010487 [-0.001418, 0.022999] |
| foundation_vs_current_cnn:94fbe1faeb95 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical | full_808 | T0-T2 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.028736 [-0.005002, 0.063609] | 0.010564 [-0.041238, 0.068827] | -0.003702 [-0.017363, 0.009329] |
| foundation_vs_current_cnn:b8d457236f29 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | -0.030803 [-0.086588, 0.022646] | -0.026155 [-0.079635, 0.027565] | 0.041190 [0.016087, 0.067342] |
| foundation_vs_current_cnn:410668394f3a | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only | full_808 | T0-T2 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.019997 [-0.074647, 0.034525] | -0.018064 [-0.065822, 0.026968] | 0.016367 [-0.005848, 0.038586] |
| foundation_vs_current_cnn:5bdc89d1a236 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.068165 [-0.132732, -0.001297] | -0.068645 [-0.150204, 0.011320] | 0.038082 [0.011243, 0.065128] |
| foundation_vs_current_cnn:c8b2f2266234 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.065146 [-0.125219, -0.002739] | -0.075305 [-0.157152, 0.011903] | 0.023913 [0.002597, 0.046123] |
| foundation_vs_current_cnn:827764acd0b5 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | 0.033276 [-0.047947, 0.111671] | 0.035912 [-0.021141, 0.091117] | 0.012104 [-0.012608, 0.037256] |
| foundation_vs_current_cnn:7d79ea68ad37 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.037736 [-0.051281, 0.124280] | 0.027301 [-0.047226, 0.098914] | -0.024150 [-0.055878, 0.007964] |
| foundation_vs_current_cnn:e64debbf9d32 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.034117 [-0.095520, 0.030445] | -0.055762 [-0.143390, 0.026717] | 0.013299 [-0.008542, 0.034754] |
| foundation_vs_current_cnn:edb59437a73d | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.022744 [-0.070786, 0.027098] | -0.038570 [-0.113426, 0.044436] | 0.008293 [-0.005777, 0.022910] |
| foundation_vs_current_cnn:6e07a060a807 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.031698 [-0.049711, 0.108810] | 0.040475 [-0.018886, 0.103394] | 0.025679 [0.000960, 0.051362] |
| foundation_vs_current_cnn:79fa38c215c9 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.004048 [-0.087090, 0.082140] | 0.011444 [-0.054152, 0.081403] | -0.014302 [-0.044178, 0.015600] |
| foundation_vs_current_cnn:07a050ef5056 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.105729 [-0.186003, -0.029263] | -0.137055 [-0.216042, -0.057705] | 0.066070 [0.031248, 0.101714] |
| foundation_vs_current_cnn:5be1d9ce09e5 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.016672 [-0.087974, 0.055799] | -0.045439 [-0.129043, 0.040780] | 0.008252 [-0.022351, 0.038712] |
| foundation_vs_current_cnn:051eac9df259 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | 0.043533 [-0.039753, 0.124365] | 0.011519 [-0.047542, 0.067544] | -0.006984 [-0.045440, 0.031227] |
| foundation_vs_current_cnn:ab76eec249dd | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.069348 [-0.020132, 0.156166] | 0.042060 [-0.038160, 0.127486] | 0.011945 [-0.018822, 0.044016] |
| foundation_vs_current_cnn:de808bfceb8e | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.101784 [-0.181713, -0.018589] | -0.091601 [-0.185010, 0.008841] | 0.018352 [-0.010568, 0.047207] |
| foundation_vs_current_cnn:8168b7932cf7 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.094288 [-0.174320, -0.010907] | -0.104985 [-0.192620, -0.011287] | 0.025534 [-0.003580, 0.054693] |
| foundation_vs_current_cnn:1c7cf75f48fe | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.094237 [0.000511, 0.186050] | 0.073673 [-0.004521, 0.157096] | -0.042536 [-0.079703, -0.005003] |
| foundation_vs_current_cnn:9978750853e5 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | 0.013465 [-0.071185, 0.099143] | 0.003096 [-0.066112, 0.076508] | 0.016112 [-0.009228, 0.042688] |
| foundation_vs_current_cnn:00e6bddbf522 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.129228 [-0.213405, -0.047525] | -0.092190 [-0.187080, -0.004264] | 0.109065 [0.063523, 0.156042] |
| foundation_vs_current_cnn:a1a29fe98078 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.016055 [-0.065319, 0.096895] | 0.023215 [-0.074498, 0.128446] | -0.029312 [-0.065408, 0.007074] |
| foundation_vs_current_cnn:1fc7c9fd3db0 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | -0.001475 [-0.088213, 0.086024] | 0.056777 [-0.018448, 0.126280] | 0.052492 [0.005875, 0.099457] |
| foundation_vs_current_cnn:ad02e6bf68dd | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.035060 [-0.050129, 0.119934] | 0.053012 [-0.024552, 0.131932] | -0.068242 [-0.111640, -0.025366] |
| foundation_vs_current_cnn:a24476dad06d | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.063396 [-0.143538, 0.016893] | -0.039358 [-0.136434, 0.066667] | 0.000975 [-0.031256, 0.033568] |
| foundation_vs_current_cnn:23ad8fef8b15 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.055317 [-0.140984, 0.033636] | -0.061277 [-0.149071, 0.032238] | 0.010606 [-0.031725, 0.052021] |
| foundation_vs_current_cnn:4c2226a26c19 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.000240 [-0.091013, 0.089211] | 0.031122 [-0.039270, 0.106594] | -0.039009 [-0.076392, -0.001507] |
| foundation_vs_current_cnn:13800de74904 | foundation_vs_current_cnn | foundation_minus_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.000412 [-0.084802, 0.086225] | 0.006738 [-0.060248, 0.084904] | -0.039388 [-0.085651, 0.005827] |
| local_vs_global:358fe24addec | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical | full_808 | T0 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.032236 [-0.059608, -0.004940] | -0.037236 [-0.082704, 0.008771] | 0.007189 [-0.000849, 0.015106] |
| local_vs_global:03922cfca312 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only | full_808 | T0 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | 0.014794 [-0.040058, 0.069751] | 0.024666 [-0.017103, 0.070043] | 0.029628 [0.012805, 0.046094] |
| local_vs_global:e8d16fde1b5b | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.034228 [-0.064803, -0.004038] | -0.031105 [-0.078783, 0.014132] | 0.009265 [-0.000782, 0.019726] |
| local_vs_global:40c12b6a05f1 | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | -0.008030 [-0.062427, 0.044132] | -0.006608 [-0.058130, 0.046576] | 0.023857 [0.000233, 0.048163] |
| local_vs_global:9793e3509f11 | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.039990 [0.004896, 0.078997] | 0.026811 [-0.027344, 0.081950] | -0.016785 [-0.031950, -0.002342] |
| local_vs_global:5ce9df27b087 | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.011643 [-0.063230, 0.039828] | -0.012050 [-0.060743, 0.036355] | -0.010145 [-0.033358, 0.014183] |
| local_vs_global:e32f551496f5 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.031076 [-0.062061, 0.000010] | -0.038445 [-0.090418, 0.012972] | 0.010726 [0.001003, 0.020651] |
| local_vs_global:f6dbf8dfc917 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | 0.046468 [-0.012145, 0.103958] | 0.043235 [-0.009487, 0.096440] | -0.001625 [-0.019675, 0.015918] |
| local_vs_global:e513094eb643 | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.031267 [-0.061812, -0.002758] | -0.056316 [-0.104035, -0.011960] | 0.003143 [-0.005474, 0.011972] |
| local_vs_global:2149d1c7179d | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.006795 [-0.045401, 0.059726] | -0.003689 [-0.054327, 0.046213] | -0.024673 [-0.045976, -0.004230] |
| local_vs_global:ee568cf9d68f | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0-T1 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.003940 [-0.004722, 0.012575] | 0.005130 [-0.017761, 0.026937] | -0.000328 [-0.001963, 0.001506] |
| local_vs_global:92f579ee49c4 | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0-T1 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.028668 [-0.080589, 0.022327] | -0.027093 [-0.076140, 0.017925] | 0.008800 [-0.011149, 0.029093] |
| local_vs_global:8f78e01e0ab1 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.021832 [-0.061785, 0.016921] | -0.033769 [-0.091485, 0.018271] | 0.010630 [-0.003506, 0.025142] |
| local_vs_global:996bb8dad90e | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | -0.005042 [-0.062287, 0.050799] | -0.022608 [-0.074041, 0.030533] | -0.002086 [-0.022532, 0.019116] |
| local_vs_global:53825301a6af | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.034849 [-0.068231, -0.003398] | -0.043448 [-0.094651, 0.006638] | 0.002457 [-0.006800, 0.011865] |
| local_vs_global:ecdb3f901c81 | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.038479 [-0.015519, 0.093543] | 0.020740 [-0.035872, 0.082045] | -0.045966 [-0.070156, -0.022257] |
| local_vs_global:3865920b67ed | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical | full_808 | T0-T2 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.029275 [0.000429, 0.058344] | 0.029790 [-0.018279, 0.078022] | -0.003559 [-0.013019, 0.006106] |
| local_vs_global:07ea18deb5eb | local_vs_global | LOCAL_minus_GLOBAL_mri_only | full_808 | T0-T2 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | 0.005765 [-0.046234, 0.058078] | -0.014517 [-0.059887, 0.029276] | -0.026909 [-0.051773, -0.002636] |
| local_vs_global:af99670f7739 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | 0.014820 [-0.033049, 0.065683] | -0.001243 [-0.058939, 0.060792] | 0.005402 [-0.010372, 0.020317] |
| local_vs_global:5b20e2c64289 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.019760 [-0.063340, 0.103352] | 0.023262 [-0.033035, 0.086799] | 0.034202 [0.007018, 0.061511] |
| local_vs_global:2bd2e98ff9c0 | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.017839 [-0.047337, 0.084737] | -0.007903 [-0.084150, 0.075593] | -0.008767 [-0.039259, 0.020942] |
| local_vs_global:3625755121ab | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.024220 [-0.052222, 0.103503] | 0.014651 [-0.052572, 0.084566] | -0.002052 [-0.031108, 0.026147] |
| local_vs_global:234d92489615 | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | 0.026192 [-0.025720, 0.078654] | 0.015949 [-0.051930, 0.095072] | 0.000396 [-0.017037, 0.017254] |
| local_vs_global:240a28bf8f86 | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.015986 [-0.085554, 0.058040] | -0.005770 [-0.067362, 0.059814] | -0.005779 [-0.030608, 0.017426] |
| local_vs_global:d2446fedc165 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.011938 [-0.080054, 0.058687] | -0.021496 [-0.108781, 0.061511] | 0.006076 [-0.019815, 0.031220] |
| local_vs_global:fdbf1453aec5 | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.041595 [-0.048172, 0.130178] | 0.030655 [-0.037236, 0.093248] | -0.050020 [-0.082580, -0.018185] |
| local_vs_global:c99dd8ec198d | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.077118 [0.000864, 0.153803] | 0.070120 [-0.004616, 0.145087] | -0.051743 [-0.090074, -0.014134] |
| local_vs_global:6d4a037ea422 | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.067410 [-0.016511, 0.149327] | 0.061196 [-0.004128, 0.134829] | -0.031091 [-0.068514, 0.008321] |
| local_vs_global:d8341bcb8301 | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0-T1 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.004443 [-0.086625, 0.080642] | -0.034880 [-0.121723, 0.050015] | 0.013259 [-0.015382, 0.042108] |
| local_vs_global:2a841891335f | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0-T1 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.039177 [-0.112514, 0.036658] | -0.039922 [-0.109456, 0.027275] | 0.008629 [-0.019055, 0.035495] |
| local_vs_global:479ea81f5b0d | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.036398 [-0.108152, 0.037673] | -0.057038 [-0.138995, 0.025810] | 0.029520 [-0.004373, 0.062926] |
| local_vs_global:50b4c4520edd | local_vs_global | LOCAL_minus_GLOBAL_current_CNN_mri_only_paired | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.033242 [-0.052678, 0.118840] | 0.025215 [-0.035187, 0.084415] | 0.030224 [-0.014280, 0.075012] |
| local_vs_global:fa7818f17f6f | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.108885 [0.022748, 0.194800] | 0.058367 [-0.031124, 0.159759] | -0.108857 [-0.154478, -0.064838] |
| local_vs_global:52a2f48e9e3d | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.069777 [-0.018062, 0.159538] | 0.021449 [-0.067144, 0.114442] | -0.090510 [-0.134919, -0.046858] |
| local_vs_global:77c142e291ab | local_vs_global | LOCAL_minus_GLOBAL_mri_clinical_paired | radiomics_complete_case_375 | T0-T2 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.028319 [-0.099248, 0.042437] | -0.078957 [-0.164059, 0.002983] | 0.039151 [0.006881, 0.071171] |
| local_vs_global:2098227cbfb4 | local_vs_global | LOCAL_minus_GLOBAL_mri_only_paired | radiomics_complete_case_375 | T0-T2 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | 0.032590 [-0.043333, 0.109605] | 0.000830 [-0.065831, 0.068490] | 0.029845 [-0.003275, 0.062537] |

## 7. HR/HER2 phenotype probes

| Target | Model | Spatial | Timing | n | Positive | AUROC | AUPRC | Brier | Calibration intercept | Calibration slope | ECE10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HER2 | GAP0 | GLOBAL | T0 | 808 | 201 | 0.546911 | 0.284369 | 0.254607 | -1.090696 | 0.147469 | 0.243761 |
| HER2 | LOCAL0 | LOCAL | T0 | 808 | 201 | 0.525109 | 0.264344 | 0.274497 | -1.100480 | 0.023307 | 0.256396 |
| HER2 | dino_vitb16_imagenet1k | GLOBAL | T0 | 808 | 201 | 0.522995 | 0.263192 | 0.269111 | 3.740873 | 1.000000 | 0.258168 |
| HER2 | dino_vitb16_imagenet1k | LOCAL | T0 | 808 | 201 | 0.556935 | 0.298620 | 0.253384 | -1.093343 | 0.301531 | 0.236531 |
| HER2 | medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 808 | 201 | 0.519183 | 0.273761 | 0.250496 | -1.105811 | -0.013281 | 0.245555 |
| HER2 | medicalnet_resnet50_3dseg8 | LOCAL | T0 | 808 | 201 | 0.463785 | 0.227163 | 0.278634 | 2.004089 | 1.000000 | 0.267472 |
| HR | GAP0 | GLOBAL | T0 | 808 | 453 | 0.537966 | 0.581418 | 0.253640 | 0.239568 | 0.193713 | 0.074673 |
| HR | LOCAL0 | LOCAL | T0 | 808 | 453 | 0.500246 | 0.564081 | 0.268041 | 0.243934 | -0.006683 | 0.107092 |
| HR | dino_vitb16_imagenet1k | GLOBAL | T0 | 808 | 453 | 0.521686 | 0.582638 | 0.283054 | -3.409462 | 1.000000 | 0.131050 |
| HR | dino_vitb16_imagenet1k | LOCAL | T0 | 808 | 453 | 0.596257 | 0.629374 | 0.273781 | -2.362360 | 3.014509 | 0.154211 |
| HR | medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 808 | 453 | 0.487765 | 0.538504 | 0.295836 | -2.231358 | 2.837636 | 0.166464 |
| HR | medicalnet_resnet50_3dseg8 | LOCAL | T0 | 808 | 453 | 0.534782 | 0.583168 | 0.260268 | -2.040053 | 3.407448 | 0.098592 |

## 8. 四分类 HR/HER2 subtype probe

Subtype 使用 macro one-vs-rest AUROC/AUPRC、multiclass Brier 与 top-label ECE；不能把 binary calibration slope/intercept 套用于本表。

| Model | Spatial | Timing | n | Macro OVR AUROC | Macro OVR AUPRC | Multiclass Brier | Top-label ECE10 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAP0 | GLOBAL | T0 | 808 | 0.536174 | 0.278729 | 0.749202 | 0.070252 | 0.303218 |
| LOCAL0 | LOCAL | T0 | 808 | 0.504645 | 0.248151 | 0.794085 | 0.139930 | 0.271040 |
| dino_vitb16_imagenet1k | GLOBAL | T0 | 808 | 0.540874 | 0.281981 | 0.783609 | 0.103462 | 0.345297 |
| dino_vitb16_imagenet1k | LOCAL | T0 | 808 | 0.564375 | 0.300734 | 0.813133 | 0.218055 | 0.339109 |
| medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 808 | 0.503792 | 0.258907 | 0.880191 | 0.215338 | 0.256188 |
| medicalnet_resnet50_3dseg8 | LOCAL | T0 | 808 | 0.496645 | 0.249732 | 0.856060 | 0.158163 | 0.290842 |

## 9. FTV 与 literal ΔFTV decodability

本表只在同一 375 complete cases 上解释。FTV 可解码不能单独证明 pCR signal 仅由 tumor size 驱动；tumor-size 以外信息的判断由完整 beyond-FTV paired 集合决定。

| Model | Spatial | Task | Endpoint | n | Spearman | Pearson | R² | RMSE | MAE | RMSE gain over b0 | Calibration slope | Calibration intercept | Mean bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAP0 | GLOBAL | delta | T0-T1 | 375 | 0.030013 | -0.022895 | -0.021226 | 25.598698 | 13.862553 | -0.004743 | -0.002850 | -11.201927 | -0.142513 |
| GAP0 | GLOBAL | delta | T1-T2 | 375 | 0.156812 | 0.146332 | -0.101857 | 18.047717 | 10.300563 | -0.048079 | 0.072630 | -8.437908 | -0.475551 |
| GAP0 | GLOBAL | delta | T2-T3 | 375 | 0.036653 | -0.033370 | -0.059902 | 28.198064 | 10.061318 | -0.023971 | -0.007109 | -5.923619 | -0.472737 |
| GAP0 | GLOBAL | static | T0 | 375 | 0.304836 | 0.290749 | 0.018069 | 42.279697 | 20.311996 | 0.011615 | 0.057119 | 16.720050 | -10.237829 |
| GAP0 | GLOBAL | static | T1 | 375 | 0.409019 | 0.306977 | 0.064406 | 36.445650 | 13.749634 | 0.039786 | 0.121158 | 9.829084 | -5.605997 |
| GAP0 | GLOBAL | static | T2 | 375 | 0.168618 | 0.123437 | -0.025716 | 31.500318 | 8.326267 | -0.004937 | 0.006548 | 3.017029 | -5.901212 |
| GAP0 | GLOBAL | static | T3 | 375 | 0.061033 | 0.043764 | -0.029251 | 12.588789 | 3.357313 | -0.008908 | 0.001841 | 1.367554 | -2.190504 |
| LOCAL0 | LOCAL | delta | T0-T1 | 375 | 0.394116 | 0.213682 | 0.038423 | 24.839840 | 13.223811 | 0.025042 | 0.063778 | -10.150787 | 0.173857 |
| LOCAL0 | LOCAL | delta | T1-T2 | 375 | 0.354320 | 0.386885 | 0.147726 | 15.872640 | 9.030576 | 0.078234 | 0.132888 | -7.588749 | -0.143763 |
| LOCAL0 | LOCAL | delta | T2-T3 | 375 | 0.171198 | 0.088723 | -0.011989 | 27.553356 | 9.720616 | -0.000559 | 0.020276 | -5.787910 | -0.485249 |
| LOCAL0 | LOCAL | static | T0 | 375 | 0.759370 | 0.612514 | 0.301816 | 35.651385 | 15.502560 | 0.166567 | 0.258159 | 13.018139 | -8.191806 |
| LOCAL0 | LOCAL | static | T1 | 375 | 0.552624 | 0.359360 | 0.093818 | 35.868212 | 11.812449 | 0.055000 | 0.102418 | 9.260649 | -6.503567 |
| LOCAL0 | LOCAL | static | T2 | 375 | 0.405494 | 0.213214 | 0.001809 | 31.074802 | 7.971825 | 0.008638 | 0.021211 | 3.335408 | -5.451202 |
| LOCAL0 | LOCAL | static | T3 | 375 | 0.317227 | 0.419947 | 0.057683 | 12.045417 | 3.180563 | 0.034639 | 0.047387 | 1.459086 | -1.936618 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T0-T1 | 375 | 0.335869 | 0.371735 | 0.116771 | 23.806391 | 15.000747 | 0.065605 | 0.192533 | -8.738960 | 0.165776 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T1-T2 | 375 | 0.296611 | 0.340175 | 0.079336 | 16.497197 | 10.303625 | 0.041964 | 0.180410 | -7.290957 | -0.253994 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T2-T3 | 375 | 0.267497 | 0.325123 | 0.082986 | 26.228559 | 12.215286 | 0.047549 | 0.154631 | -4.341574 | 0.233907 |
| dino_vitb16_imagenet1k | GLOBAL | static | T0 | 375 | 0.620154 | 0.677530 | 0.395417 | 33.175627 | 15.996101 | 0.224444 | 0.326110 | 12.503195 | -6.763977 |
| dino_vitb16_imagenet1k | GLOBAL | static | T1 | 375 | 0.573859 | 0.659448 | 0.398824 | 29.214792 | 10.631295 | 0.230294 | 0.361920 | 5.392577 | -5.814005 |
| dino_vitb16_imagenet1k | GLOBAL | static | T2 | 375 | 0.522988 | 0.824163 | 0.325547 | 25.543315 | 6.914659 | 0.185106 | 0.203806 | 2.649515 | -4.497939 |
| dino_vitb16_imagenet1k | GLOBAL | static | T3 | 375 | 0.347098 | 0.442226 | 0.065280 | 11.996766 | 3.212831 | 0.038539 | 0.050441 | 1.519720 | -1.865099 |
| dino_vitb16_imagenet1k | LOCAL | delta | T0-T1 | 375 | 0.507507 | 0.478044 | 0.224963 | 22.300680 | 12.451312 | 0.124704 | 0.257051 | -8.149342 | 0.043889 |
| dino_vitb16_imagenet1k | LOCAL | delta | T1-T2 | 375 | 0.487911 | 0.483935 | 0.224830 | 15.137641 | 8.847370 | 0.120917 | 0.280703 | -6.369507 | -0.193652 |
| dino_vitb16_imagenet1k | LOCAL | delta | T2-T3 | 375 | 0.287688 | 0.404343 | 0.152185 | 25.219533 | 11.725114 | 0.084190 | 0.206432 | -4.142865 | 0.152244 |
| dino_vitb16_imagenet1k | LOCAL | static | T0 | 375 | 0.834256 | 0.792818 | 0.623395 | 26.183916 | 10.548841 | 0.387891 | 0.620262 | 7.823301 | -3.033778 |
| dino_vitb16_imagenet1k | LOCAL | static | T1 | 375 | 0.725555 | 0.629651 | 0.322910 | 31.004547 | 9.618429 | 0.183140 | 0.249464 | 7.981267 | -5.200381 |
| dino_vitb16_imagenet1k | LOCAL | static | T2 | 375 | 0.576348 | 0.599943 | 0.154602 | 28.597755 | 6.954511 | 0.087662 | 0.104639 | 3.193727 | -4.843953 |
| dino_vitb16_imagenet1k | LOCAL | static | T3 | 375 | 0.373247 | 0.479004 | 0.112936 | 11.686939 | 3.168534 | 0.063369 | 0.080253 | 1.545771 | -1.732777 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T0-T1 | 375 | 0.237314 | 0.022123 | -8.559987 | 78.322314 | 25.155601 | -2.074132 | 0.065135 | -14.075689 | -3.766015 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T1-T2 | 375 | 0.327088 | 0.028945 | -10.683688 | 58.769175 | 16.914675 | -2.412882 | 0.095214 | -11.755356 | -3.986902 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T2-T3 | 375 | 0.207300 | 0.077409 | -5.920406 | 72.053037 | 22.548396 | -1.616499 | 0.194397 | -5.751722 | -1.391475 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T0 | 375 | 0.515153 | 0.148424 | -1.115669 | 62.060484 | 23.639317 | -0.450806 | 0.180330 | 22.847222 | -0.587930 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T1 | 375 | 0.496834 | 0.069384 | -1.086708 | 54.429328 | 16.433852 | -0.434020 | 0.077063 | 13.006642 | -3.202887 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T2 | 375 | 0.253130 | 0.016149 | -0.973635 | 43.695316 | 12.040042 | -0.393987 | 0.016179 | 7.337155 | -1.494630 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T3 | 375 | 0.135276 | 0.010579 | -1.735664 | 20.523655 | 5.224511 | -0.644835 | 0.014050 | 3.437736 | -0.076804 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T0-T1 | 375 | 0.140221 | 0.009287 | -516.791977 | 576.413987 | 90.319048 | -21.624109 | 0.211201 | -5.068171 | 3.630691 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T1-T2 | 375 | 0.360476 | 0.018365 | -271882.473651 | 8965.014945 | 494.353845 | -519.622212 | 9.563576 | 533.959547 | 460.433058 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T2-T3 | 375 | 0.165208 | -0.008735 | -82687.431842 | 7876.057545 | 457.864038 | -285.007330 | -2.508056 | -445.617605 | -426.630587 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T0 | 375 | 0.563681 | 0.050988 | -3.456286 | 90.069523 | 33.146843 | -1.105581 | 0.096844 | 34.605428 | 8.783333 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T1 | 375 | 0.341747 | 0.024534 | -3.685830 | 81.563410 | 25.629018 | -1.148907 | 0.047563 | 22.400245 | 5.672605 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T2 | 375 | 0.140690 | -0.017557 | -3.493030 | 65.928180 | 18.069928 | -1.103270 | -0.032415 | 13.595722 | 4.327707 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T3 | 375 | 0.153302 | 0.001603 | -3.008501 | 24.843591 | 6.594372 | -0.991050 | 0.002777 | 4.878419 | 1.323698 |

完整自动汇总另见 [results summary](results_summary.md)，timing 与 calibration 图分别见 [pCR timing figure](../figures/pcr_timing_performance.png) 与 [calibration/complementarity figure](../figures/calibration_clinical_complementarity.png)。

## 10. Mixed producer lineage

| Stage | Version | Lock/code SHA | Receipt | argv SHA | Artifact SHA |
| --- | --- | --- | --- | --- | --- |
| baseline_v2 | v2 | b15f7023b7021f5c1169b51cf6bc8fe0cc1d9085102a61fbdb1d68589fe2edc5 | 8fb79e1e4316e2fab94e386c3e2a0bbebde074d78771341188818e2d9702f42c | 755957f2143829fb27dc2317e35d1490dd2d2d630ff98ce06a4ad8691a8a1864 | metrics=9ba743cfb6515784bd06d1384c1fe3bcad1d2d9873e00ba438bed1e31b71f52e, predictions=d45b55341fc8c6c0c169d547a3b47d81dc55d9423e0e06be0be72bea654b350e, progress=4e52fe3f379f7e1f25b5892a4ea2d66cd3f00657e058c2ec282c100e7fb1180a, selection=313a7b38e413381adeceee3fbe40f0baf0cd086377346e83de8620b1e010b937 |
| probe_v3 | v3 | 8e8c4a5488fc862e1c73ac643495216bcc6eb015b767d7be39150894c8265104 | c7b515a099a5da6fb2a88e25cad92ddb6aaa45d2fb8b9a79e7357b4381955505 | 0d1af3956391f7c58a48eaffae914c0af51b1eceffb5ba792624d2e04a4923ec | ftv_metrics=0341cffefe45e5e393ff9dcad147c5839331bf32608cd8ade3b2ca49d4bb76a8, ftv_predictions=36b9b29d51cdde788cdf81193ce670906bf7231e08edc65e4f373c6e1f042f69, ftv_selection=980dc29507f532d3088226f8b6fc7c3383e49bfde5365e0cc399294e49a7aaaf, phenotype_metrics=042258b147c6076083846aad80c9df71c0e1236b51573410b4025505cd5c93f0, phenotype_predictions=356705b0e3e21988579d0b1f30242741f57c3d6a42619a71b6da4928822aa877, phenotype_selection=f5107a2c295b67a15652c3243a6bfc86539b9665e20686daa7f934b28e13ce85, progress=07c21dea7e06ecf8805809ad5fe13cffa1608fdee0e191d50d63454468a078eb, subtype_metrics=c05256803c629122625078d4f1c9f7eed0974554fcf86720e4d2e17f24162279, subtype_predictions=f80b571dcf121729910e14290cb47d73ddd3e7f5b928f111d48ab8263a818010, subtype_selection=4f949691bb6b3b64cda0fb10b7b2da613bc188fb419edd8ae7f5e357e29b916d |
| reporting_v3 | v3 | 5be484747ac6aeb5b622ecb70f2590e214f687e714e6d111a73f4ee775b165ca | reporting_run_provenance commit marker | af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc | calibration_figure=31c3e7ea14ba00a6ed862049fdb48b3115c50be7da192eb91036d8a1a70648b5, paired_public=76a55e828c024513d5fcbe5ed9db4e2c3768d1d5657fe71cea13be316c08c363, results_summary_markdown=646d21d815bef09695230a0a4ae6dd153604c810b6f23abbe0fbd69ba294697d, summary_json=bec3af0bd2b73695f5933cce25fb17b8dd2c72589be3784196e5e5f9605575b8, timing_figure=0c0accec1b7a9fda27a59cfedb57870346afa456384f8a6c8357b52a62192df0 |

## 11. 失败与执行谱系边界

- v1 baseline 与 probe 均以 exit code 143 终止，未产生可复用或被查看的正式 prediction/metric artifacts；但已在内存处理部分 outcome/test prediction，因此不能写成“未运行 test”。
- 初版 multinomial-SAGA outcome-blind runtime smoke 在预声明边界以 exit code 143 终止；最终显式四分类 one-vs-rest liblinear smoke 通过。
- probe-v2 正式运行以 exit code 1 失败，没有写出 prediction、selection 或 public metric artifact。原 traceback 未保留；metric-free 日志窗口与完全 synthetic、outcome-blind 复现高度匹配于 static-FTV transformed Ridge prediction 经 `expm1` 溢出后的 nonfinite outer-test prediction。这是最可能根因，不应写成已恢复原 traceback的确定事实。
- 最终发布必须在 execution ledger 中同时保留 v1、v2 与最终成功版本的 lock、receipt、artifact SHA 和真实状态；不得用最终成功记录覆盖历史失败。

## 12. 限制

1. 本轮是 frozen encoder 加线性/Ridge probe，不是 fine-tuning 上限。
2. 没有外部 cohort；bootstrap 仅描述本 cohort 的 OOF prediction 不确定性。
3. 375 人 FTV/radiomics complete-case 缺失非随机，绝对指标不得与 808 人直接排名。
4. LOCAL 继承冻结的 T0 localisation prior，不能描述为完全无定位先验。
5. DINO 是固定 2-D axial adaptation，不代表原生 3-D architecture。
6. MedicalNet 上游未发布 checksum；可重复性属于取得时 hash 与 strict-load 的 conditional pass。
7. 所有多模型、多时点区间均为未作 multiplicity adjustment 的描述性结果。
8. 区间跨 0 不证明严格等效；区间未跨 0 也不作确认性显著性声明。
9. Report 与 coverage receipt 分别跨目录发布，不能在 SIGKILL 下构成 filesystem transaction；coverage receipt 是唯一 publication commit marker。若只存在其中一个，下一次运行 fail closed 并要求 operator audit，不自动恢复。
10. AUROC/AUPRC/Brier/ECE 与定义上必有值的误差指标必须 finite 并满足固定域；constant prediction、constant target 或 `b0_rmse=0` 时定义上不可用的 calibration、correlation 或 RMSE gain 以 `NA` 原样保留。任何 infinity 均拒绝，JSON `null` 必须与同 identity 的 CSV `NaN` 对齐。

## 13. 公开输入 provenance

| Public input | Relative link | SHA-256 |
| --- | --- | --- |
| baseline_public | [baseline_public](../metrics/baseline_metrics.csv) | 9ba743cfb6515784bd06d1384c1fe3bcad1d2d9873e00ba438bed1e31b71f52e |
| calibration_figure | [calibration_figure](../figures/calibration_clinical_complementarity.png) | 31c3e7ea14ba00a6ed862049fdb48b3115c50be7da192eb91036d8a1a70648b5 |
| contract_json | [contract_json](../configs/final_report_contract.json) | 79b07f5de2d6118ba02fc0aa8b0e1f51f8bfe8b7ced0833acb1fa75e4ccd590b |
| current_cnn_provenance_audit | [current_cnn_provenance_audit](current_cnn_provenance_audit.md) | fc091467cb0254963b04f84be80653afddcecdfd83ee366b994d646b9e3da9ad |
| finalization_lock | [finalization_lock](../configs/FINALIZATION_LOCK.v1.json) | 30c0c0e6ce7d92fb6164368addf467e2142e5159fb95240f4c030ddb986c4e7b |
| finalization_module | [finalization_module](../src/foundation_mri/finalization.py) | 4ce4c0109bbf5133d0ad109f3ab71f321a83c2cdbb46dfc0e70fffffd69770d3 |
| finalization_test | [finalization_test](../tests/test_finalization.py) | e76e82ff947df965b0fee5593ce9fc9ff7e4d4ba422f133af81d92511849b085 |
| finalizer_cli | [finalizer_cli](../scripts/finalize_report.py) | 4f9d8bd0c5f1689c754f5c42a96a15b88a3ae7b7803229c18c32d39641a72293 |
| foundation_model_selection | [foundation_model_selection](foundation_model_selection.md) | 77fe49a82aadcc988a5a937e2190df8774f5992ec85bd68a9a7c1d113bd58d08 |
| ftv_public | [ftv_public](../metrics/ftv_probe_metrics.csv) | 0341cffefe45e5e393ff9dcad147c5839331bf32608cd8ade3b2ca49d4bb76a8 |
| git_handoff_json | [git_handoff_json](git_handoff.json) | 6f6a484e77a18389e4c4833769ec32f165ab2ed8d3e9f9395dfc2f4434efa2fd |
| model_execution_ledger | [model_execution_ledger](model_execution_ledger.md) | 302d4dd8426d18f7d3f0fbe309bcc8854484c81360a391e9695279388e78523e |
| paired_public | [paired_public](../metrics/paired_bootstrap_comparisons.csv) | 76a55e828c024513d5fcbe5ed9db4e2c3768d1d5657fe71cea13be316c08c363 |
| phenotype_public | [phenotype_public](../metrics/phenotype_metrics.csv) | 042258b147c6076083846aad80c9df71c0e1236b51573410b4025505cd5c93f0 |
| reporting_run_provenance | [reporting_run_provenance](../metrics/reporting_run_provenance.json) | ea917e3adfb8391fc396d463eb6aa1ed1bd73e0f7182fe0ebcb0a301c487e325 |
| results_summary_markdown | [results_summary_markdown](results_summary.md) | 646d21d815bef09695230a0a4ae6dd153604c810b6f23abbe0fbd69ba294697d |
| subtype_public | [subtype_public](../metrics/subtype_metrics.csv) | c05256803c629122625078d4f1c9f7eed0974554fcf86720e4d2e17f24162279 |
| summary_json | [summary_json](../metrics/results_summary.json) | bec3af0bd2b73695f5933cce25fb17b8dd2c72589be3784196e5e5f9605575b8 |
| template_markdown | [template_markdown](final_report.template.md) | 3ce8006aee4e3ead4e91cbe9663faf39d0937a29f8a5b0af51ea132d0d49ef0c |
| timing_figure | [timing_figure](../figures/pcr_timing_performance.png) | 0c0accec1b7a9fda27a59cfedb57870346afa456384f8a6c8357b52a62192df0 |

公开 coverage receipt 另行记录 canonical identity digests、question subset counts、固定结论分支和本报告 SHA。Patient-level features、predictions、selection rows、checkpoints、clinical/radiomics source data 与运行日志的 bytes/content/rows 不进入本报告或 Git；mixed-lineage 表只保留 generic artifact role 与不可逆 SHA-256。

## 14. Git handoff

| Item | Substantive content push value |
| --- | --- |
| Content commit | 42c5a882479d0852a88e3e960f94322dc1dc7fdc |
| Branch | feature/foundation-mri-baselines |
| Attempted remote/ref | origin refs/heads/feature/foundation-mri-baselines |
| substantive_push_status | SUBSTANTIVE_PUSH_OK |
| substantive_remote_ref_sha | 42c5a882479d0852a88e3e960f94322dc1dc7fdc |
| Sanitized push error | 无 |

上表只记录首次 substantive content push。报告与 coverage receipt 随后的 metadata push 状态不能自引用写回 receipt-bound report；无论第二次 push 成功或失败，都只能在最终交付消息中准确报告，失败后不得再次修改本报告来追写状态。
