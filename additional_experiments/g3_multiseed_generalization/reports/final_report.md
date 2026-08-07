# G3 多训练种子泛化最终报告

## 结论先行

预注册机械结论为 **PROMISING BUT UNSTABLE**。R1/R2/R3/R4 分别为
`True` / `True` /
`False` / `True`。
pCR 是次要终点，机器字段固定为 `pcr_used_in_decision=false`，没有改变正式结论。

## 五个训练种子的主结果

| 训练种子 seed_base | dS | dD | dD_R2 | pCR 纵向 ΔAUROC | 失败 fold 数 |
|---:|---:|---:|---:|---:|---:|
| 2026 | +0.0674 | +0.0760 | +0.0546 | +0.0025 | 1 |
| 3026 | +0.0701 | +0.0803 | +0.0558 | -0.0005 | 1 |
| 4026 | +0.0453 | +0.0877 | +0.0662 | -0.0059 | 0 |
| 5026 | +0.0610 | +0.0833 | +0.0533 | +0.0040 | 3 |
| 6026 | +0.0420 | +0.1199 | +0.0724 | +0.0235 | 3 |

- 静态主效应均值 dS=+0.0572，样本标准差=0.0128，
  训练种子层面 95% t-CI [+0.0412, +0.0731]。
- 动态主效应均值 dD=+0.0894，样本标准差=0.0176，
  训练种子层面 95% t-CI [+0.0676, +0.1112]。
- 支持性的交叉 bootstrap 区间：dS [+0.0353, +0.0772]，
  dD [+0.0618, +0.1186]；它们不进入正式门槛。

## pCR 次要终点

下表是五个训练种子的合并 OOF 指标再取跨种子均值；每个种子、模型和决策点均覆盖 808 名唯一患者。

| 模型 | 决策点 | AUROC | AUPRC | 准确率 | 敏感度 | 特异度 |
|---|---|---:|---:|---:|---:|---:|
| G1 | T0 | 0.5131 | 0.3543 | 0.4923 | 0.5447 | 0.4653 |
| G1 | T0-T1 | 0.5483 | 0.3787 | 0.5267 | 0.5069 | 0.5370 |
| G1 | T0-T2 | 0.5382 | 0.3705 | 0.5337 | 0.5135 | 0.5441 |
| G3 | T0 | 0.5176 | 0.3541 | 0.5391 | 0.4487 | 0.5857 |
| G3 | T0-T1 | 0.5416 | 0.3811 | 0.5502 | 0.4575 | 0.5981 |
| G3 | T0-T2 | 0.5544 | 0.3835 | 0.5547 | 0.4858 | 0.5902 |

配对的 G3−G1 差值如下；AUROC、AUPRC 及其 2,000 次患者配对 bootstrap 只提供次要证据，不参与 R1–R4 或三级结论。

| 决策点 | ΔAUROC 均值±样本标准差 | 最小/中位/最大 | 95% t-CI | ΔAUPRC 均值±样本标准差 | 最小/中位/最大 | 95% t-CI |
|---|---:|---:|---:|---:|---:|---:|
| T0 | +0.0046±0.0130 | -0.0107/-0.0000/+0.0229 | [-0.0115, +0.0206] | -0.0001±0.0144 | -0.0226/+0.0010/+0.0147 | [-0.0181, +0.0178] |
| T0-T1 | -0.0067±0.0129 | -0.0212/-0.0009/+0.0065 | [-0.0227, +0.0093] | +0.0024±0.0049 | -0.0055/+0.0042/+0.0072 | [-0.0037, +0.0084] |
| T0-T2 | +0.0162±0.0287 | -0.0015/+0.0063/+0.0668 | [-0.0194, +0.0518] | +0.0130±0.0204 | -0.0051/+0.0083/+0.0478 | [-0.0123, +0.0383] |
| 纵向宏平均（T0-T1/T0-T2） | +0.0047±0.0112 | -0.0059/+0.0025/+0.0235 | [-0.0091, +0.0186] | +0.0077±0.0106 | +0.0010/+0.0037/+0.0262 | [-0.0054, +0.0208] |

逐训练种子、模型、决策点的 AUROC/AUPRC/准确率/敏感度/特异度见
[pCR 模型指标表](../metrics/final/pcr_seed_model_metrics.csv)，配对差值见
[pCR 次要终点表](../metrics/final/pcr_secondary_seed_metrics.csv) 与
[条件 bootstrap 表](../metrics/final/conditional_seed_bootstrap_ci.csv)。

## 七个冻结问题的回答

1. **静态 FTV 改善是否跨训练种子可重复？** R1=True；
   依据为 [训练种子层面端点](../metrics/final/seed_level_robustness.csv)、
   [训练种子 t-CI](../metrics/final/seed_uncertainty.csv) 与
   [逐一剔除种子/fold 重算](../metrics/final/leave_one_out_sensitivity.csv)。
2. **观测 ΔFTV 改善是否跨训练种子可重复？** R2=True；
   dD 和描述性 dD_R2 均从每个训练种子五折合并后的 375 名唯一 OOF 患者重算。
3. **上一轮 fold 3 失败是否重复？** 本轮 fold 3 有 2/5 个基础损失失败，
   预注册解释为“少数重复、提示冲突但不足以称系统”。上一轮 +9.5934% 只作外部参考。
4. **不稳定性主要来自哪里？** 动态双因素方差分解的主导标签为
   `interaction_and_metric_sampling_error`；训练种子/fold/交互+采样误差的截断后占比分别为
   0.000/0.124/
   0.876。由于每格没有重复，残差不解释为纯交互。
5. **正式类别？** **PROMISING BUT UNSTABLE**，完全由未四舍五入机器表机械得到。
6. **是否值得作为 Factorized Grounded Response State 基础？** 暂不应直接作为下一阶段基础。
7. **下一步应扩展监督目标，还是先解决优化问题？** 应先解决优化与基础损失稳定性，再讨论扩展结构化监督目标。

## 统计与审计边界

- 正式主不确定性是 5 个训练种子的 t-CI；每个种子的 2,000 次患者 bootstrap 只条件于已拟合模型。
- 5,000 次交叉 bootstrap 同时重采训练种子，并在外层 fold 内同步重采患者，仅作敏感性分析。
- 逐一剔除 fold 的分析会删除相应患者行后重新计算合并 Spearman，未使用 fold rho 的代数平均。
- 公开 `metrics/final` 只有聚合表，不含患者 ID；患者级预测、特征、checkpoint 和训练历史不进入公开结果。
- 完整机器判定见 [decision.json](../metrics/final/decision.json)，输入与图像哈希见各 manifest。

## 注册图

1. [静态端点的训练种子条件区间](../figures/final/01_static_seed_conditional_ci.png)
2. [动态端点的训练种子条件区间](../figures/final/02_dynamic_seed_conditional_ci.png)
3. [基础损失退化热图](../figures/final/03_base_degradation_heatmap.png)
4. [动态增益热图](../figures/final/04_dynamic_gain_heatmap.png)
5. [fold 3 基础损失退化](../figures/final/05_fold3_base_degradation.png)
6. [静态增益分布](../figures/final/06_static_gain_distribution.png)
7. [动态增益分布](../figures/final/07_dynamic_gain_distribution.png)
8. [pCR 次要终点 AUROC](../figures/final/08_pcr_secondary_auroc.png)
9. [方差分解](../figures/final/09_variance_decomposition.png)
10. [fold 层面的稳健性](../figures/final/10_fold_level_mean_sd.png)
11. [选中 epoch 与表征标准差安全性](../figures/final/11_selected_epoch_representation_std.png)
