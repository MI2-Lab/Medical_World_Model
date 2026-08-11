# Local–Global Response State Pilot 最终报告

## 结论摘要

本 pilot 的预注册结论为：

- **Scientific classification：A. LOCAL STATE VALIDATED IN PILOT**
- **最终 architecture：LOCAL**
- **Gate A（Local State Works）：PASS**
- **Gate B（Local–Global Adds Value）：FAIL**
- **Gate C（Grounding Compatibility）：PASS**
- **Gate D（Optimization Safety）：PASS，10/10 paired folds**
- **本实验不授权直接进入 FTV+LD。** 下一步应先对 LOCAL 做更大规模的 multi-seed confirmation；只有确认成功后，FTV+LD 才应以 LOCAL 为基础。

LOCAL0 相对 GAP0 的 static macro Spearman 在 seed 2026/3026 分别提高 **+0.2748/+0.2516**，observed ΔFTV macro Spearman 提高 **+0.2321/+0.1922**，static natural R² 提高 **+0.0936/+0.1086**。这说明 frozen pooling audit 中的局部优势成功进入了 end-to-end learned response state。

LG0 相对 LOCAL0 的 static macro Spearman 仅为 **+0.0063/−0.0155**，且 static natural R² 为 **−0.0224/−0.0213**。Global branch 没有提供稳定的额外价值，因此按预注册选择规则保留更简单的 LOCAL。

这些结论只能表述为 **SUPPORTED IN PILOT**。本实验只有两个 training seeds；五折是 paired sensitivity，不是五个独立训练重复，也不是统计显著性证明。

## 1. 冻结设计与执行范围

本实验只改变：

> final spatial feature map → response state

其余 C1B-H input、DCE7、technical eligibility、3-D encoder、JEPA transition、EMA target、SIGReg、optimizer、FTV target、fold、seed 和 logical-batch contract 均保持冻结。

三个 response-state architecture 为：

| Architecture | Pooling 与投影 | Grounding arms |
|---|---|---|
| GAP | 全图 GAP，128→192 | GAP0 / GAP3 |
| LOCAL | 冻结的中央 64-mm fixed physical pooling，128→192 | LOCAL0 / LOCAL3 |
| LOCAL_GLOBAL | 64-mm LOCAL 与 GAP 的原始 128-D 向量拼接，256→192 | LG0 / LG3 |

LOCAL_GLOBAL 只使用一个轻量线性投影加 LayerNorm；没有 attention、MLP、gating、mask pooling 或 lesion-conditioned pooling。正式运行时验证的 encoder final pre-GAP map 为 `128×14×22×20`，尺寸从实际 encoder 输出推导，没有在模型实现中硬编码。LOCAL 精确复用上一轮 audit 的 fractional feature-cell overlap 64-mm pooling contract，且 online/EMA target 两路完全对称。

Grounding 唯一使用 `FTV_t`，`lambda_FTV=0.25`。没有 LD、ΔFTV、pCR、clinical/treatment 或其他 morphology supervision；observed ΔFTV 只在冻结 response state 后作为 downstream probe。

正式矩阵为 `2 seeds × 5 folds × 6 arms = 60 cells`，seeds 为 2026/3026。训练严格使用 physical batch 4、gradient accumulation 8、logical B32；每个 logical batch 只做一次 clipping、optimizer step 和 EMA update。Grounded checkpoint 只在 paired state-loss 5% 安全约束内按 validation FTV loss 选择；测试集未参与 checkpoint 选择。

## 2. 评估口径

- Static probe：`r_t → FTV_t`，端点为 T0–T3。
- Dynamics probe：`r_(t+1)-r_t → FTV_(t+1)-FTV_t`，端点为 T0→T1、T1→T2、T2→T3。
- Ridge 的 scaler、target transform 和 alpha 选择均只在 outer-train/validation 内完成；每个 endpoint 对 test 仅预测一次，不 refit。
- Natural metrics 先合并五折 OOF，再按 endpoint 计算；macro 是端点的无权平均。
- Transformed-space metrics 只做五折 fold-summary，不跨折直接拼接 transformed targets。
- Primary scope 为 measurement-valid；observable-only 是 sensitivity，不替代 primary。
- 本报告中的 calibration slope 是冻结历史定义 `Cov(true,pred)/Var(true)`，即 pred-on-true 的描述性斜率，不是常见的 true-on-pred calibration slope。

## 3. 主要自然尺度结果

下表为五折 OOF 先合并后的 macro，单元格为 `Spearman / natural R²`。

| Seed | Arm | Static FTV | Observed ΔFTV |
|---:|---|---:|---:|
| 2026 | GAP0 | 0.2305 / −0.0087 | 0.0745 / −0.0610 |
| 2026 | GAP3 | 0.2672 / −0.0150 | 0.1210 / −0.0104 |
| 2026 | LOCAL0 | **0.5053 / 0.0850** | **0.3065 / 0.0581** |
| 2026 | LOCAL3 | **0.5309 / 0.0990** | **0.3402 / 0.0806** |
| 2026 | LG0 | 0.5116 / 0.0626 | 0.3441 / 0.0502 |
| 2026 | LG3 | 0.5309 / 0.1020 | 0.3739 / 0.0702 |
| 3026 | GAP0 | 0.2203 / −0.0099 | 0.0696 / −0.0018 |
| 3026 | GAP3 | 0.2551 / 0.0312 | 0.0966 / 0.0113 |
| 3026 | LOCAL0 | **0.4720 / 0.0987** | **0.2618 / 0.0206** |
| 3026 | LOCAL3 | **0.5133 / 0.1153** | **0.3002 / 0.0356** |
| 3026 | LG0 | 0.4565 / 0.0774 | 0.3028 / 0.0481 |
| 3026 | LG3 | 0.4999 / 0.0856 | 0.3499 / 0.0661 |

### 3.1 预注册 Gate A–D

| Gate | 预注册判断 | 正式观察 | 结果 |
|---|---|---|---|
| A：LOCAL works | LOCAL0−GAP0 static ρ 每 seed ≥+0.10；ΔFTV ρ 每 seed >0；static R² 不得两 seed 均下降 | static ρ **+0.2748/+0.2516**；ΔFTV ρ **+0.2321/+0.1922**；static R² **+0.0936/+0.1086** | **PASS** |
| B：LG adds value | LG0−LOCAL0 static ρ 两 seed 非负，至少一 seed ≥+0.02；R² 不得系统下降 | static ρ **+0.0063/−0.0155**；static R² **−0.0224/−0.0213** | **FAIL** |
| C：grounding compatibility | 最终候选 A3−A0 static ρ 两 seed >0；至少一 seed ΔFTV ρ ≥+0.02；R² 不得系统下降 | LOCAL3−LOCAL0 static ρ **+0.0256/+0.0413**；ΔFTV ρ **+0.0336/+0.0384**；static R² **+0.0141/+0.0166** | **PASS** |
| D：optimization safety | Grounded final candidate 至少 9/10 folds 的 state-loss degradation ≤5% | LOCAL3−LOCAL0 **10/10**；最坏 degradation **+1.269%** | **PASS** |

Machine-readable 判定见 [decision_summary.json](../metrics/decision_summary.json)。

## 4. 对 13 个问题的明确回答

### 1. LOCAL0 是否稳定优于 GAP0？

**是。** 两个 seed 的 static macro Spearman 增益为 **+0.2748/+0.2516**，observed ΔFTV 增益为 **+0.2321/+0.1922**。Static natural R² 从 GAP0 的 **−0.0087/−0.0099** 提高到 LOCAL0 的 **0.0850/0.0987**；ΔFTV natural R² 从 **−0.0610/−0.0018** 提高到 **0.0581/0.0206**。Gate A 全部条件通过。

### 2. Frozen PLOCAL advantage 是否成功转化为 end-to-end learned state？

**是，在本 pilot 的证据边界内成功转化。** 上一轮 frozen audit 中，PLOCAL 相对 GAP 的 static macro Spearman 增益为 **+0.2048/+0.1931**，恢复 legacy deficit 的 **68.1%/74.7%**；本轮 end-to-end LOCAL0 相对 GAP0 的增益进一步为 **+0.2748/+0.2516**，且 dynamics 与 natural R² 同方向改善。历史 frozen 结果只作为外部 reference；本轮的 primary matched inference 完全来自 60 个 C1B-H formal cells。历史来源见 [prospective_gates.json](../../c1b_spatial_pooling_bottleneck_audit/metrics/prospective_gates.json)。

### 3. LG0 是否稳定优于 LOCAL0？

**否。** Static macro Spearman effect 为 **+0.0063/−0.0155**：一个 seed 略正、另一个为负，且没有 seed 达到 +0.02。Static natural R² 在两个 seed 均下降 **−0.0224/−0.0213**。虽然 ΔFTV Spearman 描述性地增加 **+0.0375/+0.0410**，Gate B 并不以 dynamics 单项替代 static 条件，因此 Gate B 失败。

### 4. Global branch 是否真的有必要？

**没有证据表明必要。** Global branch 增加了参数和 response-state 方差，但没有带来可重复的 static rank 或 natural-R² 增益。按照预注册规则，LOCAL≈LG 或 LG 不稳定时选择更简单的 LOCAL，而不是因模型更复杂而保留 global branch。

### 5. FTV grounding 在 LOCAL/LG 下是否继续有效？

**是，LOCAL 与 LG 都呈正向 grounding effects；正式候选 LOCAL 的 Gate C 通过。**

- LOCAL3−LOCAL0：static Spearman **+0.0256/+0.0413**，ΔFTV Spearman **+0.0336/+0.0384**，static R² **+0.0141/+0.0166**，ΔFTV R² **+0.0226/+0.0150**。
- LG3−LG0：static Spearman **+0.0193/+0.0434**，ΔFTV Spearman **+0.0298/+0.0471**，static R² **+0.0395/+0.0082**，ΔFTV R² **+0.0200/+0.0180**。

这些是 FTV-only grounding 结果，不包含 ΔFTV 或 LD supervision。

### 6. Observed ΔFTV 是否改善？

**明显改善。** LOCAL0−GAP0 的 ΔFTV macro Spearman 增益为 **+0.2321/+0.1922**，natural R² 增益为 **+0.1190/+0.0224**。在 LOCAL 上增加 FTV grounding 后，LOCAL3−LOCAL0 又带来 Spearman **+0.0336/+0.0384** 和 R² **+0.0226/+0.0150**。这是冻结 state 的 downstream probe 证据，不是训练期间的 ΔFTV supervision，也不建立 treatment-dynamics 因果关系。

### 7. Natural-scale R² 是否改善？

**LOCAL 明确改善；LG 没有在 static 上进一步改善。** LOCAL0−GAP0 的 static R² 增益为 **+0.0936/+0.1086**，ΔFTV R² 增益为 **+0.1190/+0.0224**。LOCAL3 再相对 LOCAL0 提高 static **+0.0141/+0.0166**、ΔFTV **+0.0226/+0.0150**。相反，LG0−LOCAL0 的 static R² 为 **−0.0224/−0.0213**，是拒绝 global branch 的重要证据。

### 8. Prediction compression 是否缓解？

**部分缓解，但远未解决。** Static macro 的 prediction/target variance ratio 从 GAP0 的 **0.0375/0.0280** 提高到 LOCAL0 的 **0.0599/0.0798**，描述性 calibration slope 从 **0.0397/0.0333** 提高到 **0.0934/0.1083**；LOCAL3 slope 进一步达到 **0.1099/0.1125**。然而这些值仍远低于理想的 1。

ΔFTV 的 LOCAL0 variance ratio 为 **0.0864/0.0738**，相对 GAP0 的 **0.1024/0.0481** 并非两个 seed 都增加；但 slope 从 **0.0209/0.0232** 提高到 **0.0723/0.0473**，且 R² 从负/近零变为正。因此只能写“排序、R² 和部分校准得到改善，压缩仍显著”。LG0 的 static variance ratio 较大（**0.2868/0.1405**），但 static R² 更差，不能把较大的预测方差解释成 global branch 有效。

### 9. LOCAL3/LG3 optimization safety 如何？

**两者均为 10/10 paired folds PASS。**

- LOCAL3−LOCAL0 degradation fraction 范围为 **−3.143% 到 +1.269%**；最坏值仍远低于 +5%。
- LG3−LG0 范围为 **−8.448% 到 +1.106%**；同样全部通过。

这里的 safety 只指预注册 validation state-loss degradation contract，不等价于全面训练安全或临床安全。

### 10. 最终应该选择 GAP、LOCAL 还是 LOCAL_GLOBAL？

**选择 LOCAL。** Gate A 证明 LOCAL 明显优于 GAP；Gate B 没有证明 LOCAL_GLOBAL 稳定优于 LOCAL；Gate C/D 证明 LOCAL 的 FTV grounding 兼容且优化安全。最终分类为 **A. LOCAL STATE VALIDATED IN PILOT**。

### 11. 当前主要瓶颈是否仍然是 spatial aggregation？

**本 pilot 支持 spatial response aggregation 是关键 end-to-end bottleneck。** 在 encoder、input、transition、训练与 probe contract 都冻结时，仅将 GAP 换为固定 LOCAL pooling 就同时改善 static representation、observed dynamics 和 natural R²。由于 compression/calibration 仍明显不足，不能说 spatial aggregation 是唯一瓶颈或问题已完全解决。

### 12. 是否已经有足够证据直接进入 FTV+LD？

**否。** 本实验明确不是 FTV+LD 实验，也不授权把 LD 立即加到当前 formal run。A/C/D 的通过说明 LOCAL 值得进入 architecture confirmation，而不是说明已经完成跨阶段确认。

### 13. 如果后续可以做 FTV+LD，应基于哪个 architecture？

**条件性选择 LOCAL。** 科学顺序应为：LOCAL multi-seed confirmation → 若 static、observed dynamics 与 optimization safety 持续成立 → 以 LOCAL 开展 FTV+LD Factorized Grounding Pilot。若 confirmation 失败，则该 architecture 选择归零并重新分析，不应自动进入 FTV+LD。

## 5. Prediction compression 与剩余问题

本轮已经把 GAP0 的 near-zero/negative natural R² 提升为 LOCAL0/LOCAL3 的正 R²，也明显提高了 rank correlation 和描述性 calibration slope。与此同时，variance ratio 与 slope 仍普遍远低于 1，说明预测幅度仍被压缩。下一阶段 confirmation 应继续把 natural R²、variance ratio、calibration slope 与 Spearman 并列报告，不能只因 Spearman 提升就宣称 response representation 已完全解决。

LG 的较大 prediction variance 也揭示了一个重要区别：**更大的输出方差不等于更好的表示。** LG0 在两个 seed 的 static variance ratio 都高于 LOCAL0，但 static R² 均更差，且 Gate B 失败。因此 architecture 选择应由预注册的 rank、R² 和跨 seed 方向共同决定，而非单一方差指标。

## 6. 产物清单

### Tables

1. [Table 1 — Architecture contract](../metrics/table1_architecture_contract.csv)
2. [Table 2 — Static FTV](../metrics/table2_static_ftv.csv)
3. [Table 3 — Observed ΔFTV](../metrics/table3_observed_delta_ftv.csv)
4. [Table 4 — Paired architecture effects](../metrics/table4_paired_architecture_effects.csv)
5. [Table 5 — Grounding effects](../metrics/table5_grounding_effects.csv)
6. [Table 6 — Optimization safety](../metrics/table6_optimization_safety.csv)
7. [Table 7 — Prediction variance / calibration](../metrics/table7_prediction_variance_calibration.csv)

### Figures

1. [Local–Global architecture schematic](../figures/01_local_global_architecture_schematic.png)
2. [Static FTV Spearman comparison](../figures/02_static_ftv_spearman_comparison.png)
3. [Static FTV natural R² comparison](../figures/03_static_ftv_natural_r2_comparison.png)
4. [Observed ΔFTV Spearman comparison](../figures/04_delta_ftv_spearman_comparison.png)
5. [Observed ΔFTV natural R² comparison](../figures/05_delta_ftv_natural_r2_comparison.png)
6. [Prediction/target variance ratio](../figures/06_prediction_target_variance_ratio.png)
7. [Descriptive calibration slope](../figures/07_descriptive_calibration_slope.png)
8. [Paired fold effects](../figures/08_paired_fold_effects.png)
9. [Optimization safety heatmap](../figures/09_optimization_safety_heatmap.png)
10. [Representative training curves](../figures/10_representative_training_curves.png)

### Machine-readable summaries

- [Aggregation summary and artifact SHA-256 index](../metrics/aggregation_summary.json)
- [Decision summary](../metrics/decision_summary.json)
- [Report context](../metrics/report_context.json)
- [Natural pooled metrics](../metrics/natural_pooled_metrics.csv)
- [Paired fold effects](../metrics/paired_fold_effects.csv)
- [Transformed fold summaries](../metrics/transformed_fold_summaries.csv)
- [Training trajectories](../metrics/training_trajectories.csv)

`report_context.json` 是 aggregation 在正文撰写前生成的 machine context，因此其中 `final_report_prose_written=false` 描述的是生成时点；本文件是随后完成的正式中文正文。Aggregation summary 中的 SHA-256 索引覆盖其列出的 7 张表、辅助公开 metrics、decision/report context 与 10 张图，不包含随后写入的本报告。

## 7. Chain of custody 与验证

- Preregistration lock SHA-256：`d2fcbd6e92300debe462da1968d74f4809a03bfdccaea2cd82edfead846c4daa`
- Pilot config SHA-256：`399dfa7327aca3a4c0cc3e711c117e6e0a828b85045f7ee5953b3ab4b06f548e`
- Stage-A sentinel SHA-256：`0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb`
- Data contract SHA-256：`dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27`
- Postprocessing 60-cell chain SHA-256：`40194e1cd2d5dec99c604ab0161e5e5c4357d1633b78ba5898a7ce5e2c5ef696`

正式训练使用 3 张 GPU，60/60 cells 一次完成，运行时间 13 小时 27 分 32 秒；没有 OOM、重试或 partial resume。所有 60 个 selected checkpoints 均为 finite、experiment PASS、optimization-safety PASS。Postprocessing 完成 60 个 feature cells 和 60 个 probe cells；feature headers 全部为 `[808,4,192]` float32，且 metadata 明确记录 `ftv_head_called=false`、`test_labels_used=false`。

独立只读审计重新验证了：

- 60/60 feature→metadata→selection→checkpoint 实时 SHA 链；
- 60/60 probe output 与 postprocessing completion chain；
- `1,680` 行 probe metrics、`840` 行 Ridge selection、`62,556` 行私有 OOF prediction；
- 所有私有文件 0600、目录 0700，无 symlink、partial 或临时文件；
- 10/10 figures 可完整解码、无空白/裁切/标签重叠，且 SHA 与 aggregation index 一致；
- 最终 public-artifact privacy gate 扫描 55 个公开产物并比对 1,120 个私有 identifier denylist 值；identifier、absolute-path、unsupported-format 与 private-permission findings 均为 0；
- 冻结后的 pilot unittest：55 项通过，1 项“结果产生前必须为空”的测试因 preregistration 已冻结而按设计跳过。

公开 PNG 在 aggregation 后接受了一次无损 metadata sanitation：移除非科学性的绘图库 text metadata，并在必要时只调整连续 IDAT chunk 边界，以避免冻结二进制路径 denylist 将 URL scheme 或压缩随机字节误判为私有路径。Sanitation 前后 10/10 图像的尺寸与每个 RGBA 像素完全一致；PNG CRC、zlib 解压和 Pillow 校验均通过，更新后的 SHA-256 已写回 aggregation index。该步骤没有重绘、改变数据或修改任何统计结果。

## 8. 证据边界与后续路线

1. 两个 seed 只支持 pilot-level consistency，不支持统计证明。
2. Paired folds 是 sensitivity，不是独立 replicate。
3. Historical legacy/PLOCAL 仅为 reference line；primary comparison 是本轮 C1B-H matched runs。
4. Observed ΔFTV 是 downstream readout，不是训练监督，也不建立因果 treatment-response dynamics。
5. Gate D 只验证 selected validation state-loss 的 5% contract。
6. 本轮没有 pCR 或其他临床终点，不能外推临床获益。
7. 下一步只做 LOCAL 的更大 multi-seed confirmation；确认成功后才单独预注册 FTV+LD，不能在本实验中追加 LD、attention pooling、更强 encoder 或新的 grounding target。

最终 scientific classification 保持：

> **A. LOCAL STATE VALIDATED IN PILOT**
