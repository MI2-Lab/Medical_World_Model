# DINOv3 post-hoc 结果解释附录

本附录是在锁定汇总器已经发布完整结果之后撰写的**描述性解释**，不是预注册分析，也不改变
[`final_report.md`](final_report.md) 中由 reporting marker 封存的科学结果字节。它只把原报告的完整表格明确映射到最初的 12 个问题；不新增模型、不重新拟合、不筛选最佳 cell，也不改变原 foundation-MRI 正式实验的结论。

区间方向的统一口径为：AUROC/AUPRC 的 95% CI 全部大于 0 或 Brier 的 95% CI 全部小于 0，记为“区间有利”；反方向记为“区间不利”；跨越或接触 0 记为“不确定”。全部区间是同患者、5,000 次 bootstrap 的描述性区间，未作多重性校正。

## 12 个问题的明确回答

1. **使用了什么模型？** 本扩展只新增 `facebook/dinov3-vitb16-pretrain-lvd1689m`（ViT-B/16，85,660,416 参数）。原 DINO v1、MedicalNet、current GAP0/LOCAL0、clinical 和 radiomics 结果均保持原样，只作为哈希锁定参照。

2. **为什么选择它？** 用户在原结果公开后指定 DINOv3；B/16 与原 DINO v1 的模型尺度、patch size 和 adapter 最接近，能较公平地隔离预训练代际差异。它仍是 post-hoc sensitivity：Meta custom license 需责任人确认，LVD-1689M 没有 patient-level/I-SPY 排除清单，污染状态只能记为 unknown。

3. **MRI-only AUROC/AUPRC 是多少？** full-808 的 GLOBAL 在 T0/T0–T1/T0–T2 为 `0.528/0.362`、`0.501/0.351`、`0.551/0.383`；LOCAL 为 `0.531/0.379`、`0.605/0.436`、`0.590/0.421`。因此最好的纵向 LOCAL 点估计约为 AUROC 0.60、AUPRC 0.44，仍不是强 pCR predictor。

4. **LOCAL vs GLOBAL 如何？** DINOv3 的主要正向信号来自 LOCAL。full-808 MRI-only T0–T1 的 LOCAL−GLOBAL 为 ΔAUROC `+0.104 [0.051, 0.156]`、ΔAUPRC `+0.085 [0.031, 0.137]`、ΔBrier `−0.045 [−0.074, −0.016]`；T0–T2 的区分度区间仍跨 0，但 Brier 改善。18 个 specs、54 个指标行合计 15 个区间有利、4 个不利、35 个不确定，说明局部纵向聚合值得复验，但不是所有设计都占优。

5. **DINOv3 vs current CNN LOCAL 如何？** 没有稳定优势。全 matched-axis 的 12 个 specs、36 个指标行中，0 个区间有利、9 个不利、27 个不确定。LOCAL MRI-only 的 AUROC/AUPRC 点估计相对 LOCAL0 在 T0、T0–T1、T0–T2 分别约为 `+0.021/+0.021`、`+0.045/+0.040`、`+0.043/+0.049`，但判别区间均跨 0。三时点的 Brier 点估计均较差；其中仅 T0–T2 的区间排除 0，T0 与 T0–T1 仍不确定。

6. **HR/HER2/subtype decodability 如何？** 证据混合且没有统一升级。相对 DINO v1，DINOv3 HR AUROC 在 GLOBAL/LOCAL 为 `−0.016/−0.020`；HER2 GLOBAL 为 `+0.036`（AUPRC `+0.031`），LOCAL 为 `−0.019`。四分类 subtype 的 GLOBAL macro-AUROC `+0.020`，但 Brier `+0.147`、accuracy `−0.031`；LOCAL macro-AUROC `−0.013`。这些 probe 没有配对 CI，只能描述。

7. **FTV/ΔFTV decodability 如何？** LOCAL T0 仍能较好解码 FTV（Spearman `0.813`、R² `0.522`），但弱于 DINO v1 的 `0.834/0.623`。14 个 matched endpoints 中，Spearman 12/14 下降、R² 11/14 下降、RMSE 11/14 变差、MAE 13/14 变差；因此 DINOv3 没有带来一致的 tumor-volume representation 升级。

8. **Clinical + DINOv3 是否超过 clinical-only？** 否。12 个 specs、36 个指标行中 0 个区间有利、22 个不利、14 个不确定。full-808 T0 尚接近 clinical-only，但 T0–T1 和多个 T0–T2/complete-case 设计明显恶化；不能宣称临床互补增益。

9. **Clinical + FTV + DINOv3 是否超过 Clinical + FTV？** 否。6 个 specs 的 AUROC、AUPRC、Brier 共 18 个指标行全部区间不利。例如 T0–T2 LOCAL 为 ΔAUROC `−0.133 [−0.208, −0.057]`、ΔAUPRC `−0.147 [−0.255, −0.043]`、ΔBrier `+0.065 [0.030, 0.100]`。

10. **是否学到 tumor-size 以外、可用于 pCR 的增量信息？** 当前证据不支持。虽然 LOCAL longitudinal MRI-only 有表征信号，但 beyond-FTV 的全部 18 个指标区间均朝不利方向，不能把该信号解释为可用的 pCR 增量 phenotype。

11. **Current World Model 是否明显 underuse MRI？** 不能确认。DINOv3 相对 current CNN 的 matched 比较没有任何区间有利；相对 DINO v1 的 36 specs、108 个指标行也只有 1 个区间有利、11 个不利、96 个不确定。更合理的诊断仍是：MRI-only pCR 信号较弱，局部空间聚合可能比简单 GLOBAL pooling 更重要，但这不等于 encoder replacement 已被证明。

12. **下一步是否值得替换/增强 image encoder？** **不建议直接替换。** 值得做的是把 DINOv3-LOCAL T0–T1 作为一个明确、预注册的新假设，在独立队列或严格 nested validation 中复验，并优先测试稳健校准/受控融合；只有在许可责任人确认 custom license、污染边界可接受、且 clinical/beyond-FTV 增益能复现后，再考虑集成。原正式结论不变。

## 全矩阵方向摘要

| 比较族 | specs | 指标行 | 区间有利 | 区间不利 | 不确定 |
|---|---:|---:|---:|---:|---:|
| DINOv3 vs DINO v1 | 36 | 108 | 1 | 11 | 96 |
| DINOv3 LOCAL vs GLOBAL | 18 | 54 | 15 | 4 | 35 |
| DINOv3 vs current CNN（full） | 12 | 36 | 0 | 9 | 27 |
| Clinical gain | 12 | 36 | 0 | 22 | 14 |
| Beyond FTV | 6 | 18 | 0 | 18 | 0 |

权威数值仍以 [`results_summary.json`](../metrics/results_summary.json) 和 [`paired_bootstrap_comparisons.csv`](../metrics/paired_bootstrap_comparisons.csv) 为准。本附录使用的 source hashes 为：results summary `5d9818550cd7135ade515711e6aa54172cf840b00f9589b16a4e943dcc71b9b8`；paired comparisons `69b064f847c792c992982646bad341fa4e7e85c0df5b1cd309ae14dc1b7bad68`；scientific final report `44666a0d85e66d3f436173291fe086ef4ad684d9ee30403819f80a910e317750`；reporting provenance `e2f84967d1a050dca25273ab4154d9b085f4f03699625c611b973481c4c2f1b6`。
