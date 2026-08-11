# DINOv3 DCE-MRI post-hoc 敏感性分析

## 结论边界

本报告只增加一个固定模型：`{{MODEL_NAME}}`。它是在原 foundation-MRI 正式结果已经公开后由用户指定的 **post-hoc（事后）敏感性 baseline**，不是原预注册候选，也不构成新的确认性检验。无论本报告中的数值如何，原实验的候选集合、主要分析和既有结论均保持不变；这里的结果只能作为后续 encoder 研究的描述性证据。

DINOv3 使用 Meta 的 custom license，而不是标准开源许可证；本地实验依赖本机既有、经 hash 验证的官方 checkpoint。本实验不代表机构或 PI 已接受许可，责任人仍需确认适用条款，报告与仓库不分发权重。许可措辞的 outcome-blind 澄清见 `reports/license_scope_clarification.md`。LVD-1689M 来自不可逐项枚举的公开网络图像池，没有 patient-level 成员清单或 I-SPY 排除清单，因此 I-SPY 衍生内容的预训练污染状态为 **unknown**，不能声称为零污染。

所有候选、GLOBAL/LOCAL 轴、T0/T0-T1/T0-T2 时点以及 full/complete-case 人群均按冻结协议顺序显示；没有 best-cell filtering，也没有按 AUROC、AUPRC、Brier 或区间排序。84 个比较、252 个 metric rows 均为同患者 outer-fold OOF 配对 bootstrap 的描述性结果（seed 2026、5000 次、percentile 95% CI），不用于模型/检查点选择或确认性显著性判断。差值均为“候选减参照”；AUROC/AUPRC 越大越好，Brier 越小越好。

## 模型与公平 adapter

- 官方 checkpoint：`facebook/dinov3-vitb16-pretrain-lvd1689m`，ViT-B/16；选择 B/16 是为了与原 DINO v1 ViT-B/16 reference 尽量公平地隔离预训练代际差异。
- 输入 adapter 与原 DINO v1 baseline 一致：DCE early、late、late-minus-pre 三通道，32 个 axial slices，224×224，ImageNet normalization，并保留相同 GLOBAL 与固定 central 64-mm LOCAL 几何。
- 每 slice 的 201 个 final tokens 按固定索引处理：CLS 为 `[0]`；四个 register tokens 为 `[1:5]` 并明确排除；196 个 patch tokens 为 `[5:201]`。表示为 `CLS[0]` 与 patch mean `[5:201]` 拼接，得到 1536 维；不把 register token 混入 patch pooling。

## 覆盖与可重复性

- full cohort：{{FULL_SIZE}}
- radiomics complete-case：{{COMPLETE_SIZE}}
- DINOv3 pCR pooled cells：36
- 配对比较 specs：{{SPEC_COUNT}}
- 配对 metric rows：{{METRIC_ROW_COUNT}}
- outcome-blind comparison contract SHA-256：`{{CONTRACT_SHA256}}`
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

{{PCR_TABLE}}

## 全部 84 个配对比较（252 个 metric rows）

{{COMPARISON_TABLE}}

## HR/HER2 binary phenotype probes：DINOv3 vs DINO v1 matched public aggregates

以下 probe 比较只给 pooled aggregate 的两组绝对值与 v3−v1 差值，不计算 CI，也不进入84-spec配对 bootstrap。

{{PHENOTYPE_TABLE}}

## HR/HER2 四分类 subtype probe：DINOv3 vs DINO v1

{{SUBTYPE_TABLE}}

## FTV 与 ΔFTV decodability：DINOv3 vs DINO v1

Spearman/R² 的正差值表示 DINOv3 数值更高；RMSE/MAE 的负差值表示误差更低。这里仍是无 CI 的描述性 matched public aggregate 对照。

{{FTV_TABLE}}

## 最终解释限制

这些表和公开图完整展示所有冻结候选/axes/timings；任何看似有利或不利的单格都不得替代全矩阵判断。由于这是结果公开后的单模型敏感性扩展，且 LVD-1689M 的 patient-level contamination 不能排除，本报告不声称证明 DINOv3 的一般优势，也不改变原 foundation-MRI 正式报告的结论。

<!-- FOUNDATION_MRI_DINOV3_GIT_HANDOFF_V1 -->
