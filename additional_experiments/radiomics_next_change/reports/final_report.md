# CoRe-WM 小样本影像演化研究最终报告

日期：2026-08-06
分支：`feature/ispy-clean-corejepa`
基准 commit：`c413ec86af04795434bdc19e65bbb006c966f379`
环境：conda `bowen`；Python 3.11.14；PyTorch 2.9.1+cu130；3× NVIDIA RTX PRO 6000 Blackwell Max-Q

## 1. 执行结论

第一轮 M0、M1、M2 已按同一候选五折、seed、训练预算和冻结 readout 完成；每个模型均覆盖 808 名 I-SPY2 OOF test 患者，并保存 patient-level prediction、transition、shortcut 和 provenance。C0/C1/C2 也在同一 375 名 measurement-paired OOF test subset 上完成。

核心结论是一个有边界的负结果：

- M1 Next-Change 明显修复了 M0 在 LayerNorm latent 空间劣于 copy-current 的问题：总体 normalized aggregate transition gain 从 **-62.21%** 变为 **+6.24%**。但 raw latent gain 仍为 **-2.84%**，delta cosine 仅 0.104。
- M1 没有一致提高 pCR readout。相对 M0，T0–T1 AUROC 从 0.504 提到 0.539，但 T0 从 0.549 降到 0.516，T0–T2 从 0.549 降到 0.541。
- M2 的 λ=0.05 auxiliary supervision 几乎保持了 M1 的 transition 指标，却没有改善 ROI辅助 image-only pCR：T0/T0–T1/T0–T2 AUROC 为 0.508/0.529/0.541，相对 M1 为 -0.008/-0.009/+0.0001。
- M2 原生 radiomics head 基本预测折内均值：12 个 transition×feature 单元中 11 个被标记为 near-constant，Spearman 介于 -0.091 和 0.053；没有任何 feature 显示可复现、可靠的 patient-level grounding。
- C0 radiomics-only 在 paired subset 的 T0–T2 AUROC 达 0.756，而 C2 M2 ROI辅助 image-only 只有 0.535。这说明纵向 measurement 确有 pCR 信息，但当前 privileged auxiliary 设计没有把该信息转移到影像表征。

因此，本实验**不支持**“当前低维 radiomics privileged supervision 能改善 image-only world model 并学习更有治疗响应意义的影像演化表征”这一主张。它支持的较窄结论是：显式 Next-Change objective 能减少 normalized latent copy shortcut，但这一优化尚未转化为稳定的临床判别力或 measurement grounding。

## 2. 数据审计

### 2.1 实际 measurement 数据

权威源为 `/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx`，sheet `datawith4visits`，384×29。它不是高维 PyRadiomics 纹理表，而是四类纵向 MRI measurement：

| Feature | T0–T3 绝对值 | M2 相邻 target | 单位/变换 |
|---|---|---|---|
| FTV | `VOLUME_TUM_BLU_V10`–`V40` | log-epsilon change | cc；fold-train-only epsilon |
| Sphericity | `SPHERICITY_T0`–`T3` | absolute change | 无量纲 |
| LD | `LD_T0`–`T3` | log-epsilon change | 原表单位未明示；存在零值 |
| BPE | `BPE_5slice_mean_T0`–`T3` | absolute change | 原表单位未明示 |

每折只用 train patients 拟合 1%/99% winsorization、median/IQR、epsilon；validation/test 只应用保存参数。五个 transform JSON 均锁定 raw-target hash、train-ID hash、feature 顺序和算法版本。

### 2.2 患者重叠与 transition

| 项目 | 数量 |
|---|---:|
| 完整 T0–T3 MRI primary cohort | 808 |
| 严格六位 trial ID 匹配的 measurement 患者 | 375（46.41%） |
| 无 measurement 的 MRI 患者 | 433（53.59%） |
| 工作簿中不在 808 cohort 的患者 | 9 |
| T0→T1 / T1→T2 / T2→T3 paired patients | 各 375 |
| 相邻 patient-transitions | 1,125 |
| feature-level target cells | 4,500 |

没有使用 fuzzy matching。375 名配对患者的四类 measurement 和四访均完整；433 名未配对患者没有被删除或伪造 target，M2 中只把其 radiomics mask 置零。

### 2.3 Complete-case selection bias

measurement 可用组与不可用组不是随机可交换子集：

- pCR：29.33%（110/375）对 38.11%（165/433），差 -8.77 个百分点；
- T0 ROI mask 体素数中位数：16,927 对 12,991；
- HR+/HER2-：42.13% 对 37.41%；
- `targeted_other` treatment family：41.60% 对 34.18%；
- 年龄：48.26±10.14 对 49.46±10.58 岁。

因此 paired 控制只能在相同 375 人内解释，不能把其 AUROC 直接外推到 808 人。完整审计见 [radiomics_data_audit.md](radiomics_data_audit.md)。

## 3. 仓库复现边界

clean 分支没有现成的 `_corejepa_clean_dce8` cache、五折 checkpoint、frozen state 或 FLR 结果；原生代码也只有一次 70/15/15 split，而不是当前要求的五折。找到的 seed-2026 manifest 结构正确：808 人在每个 fold 恰好一次 test，SHA-256 为 `143e482d...aa38`，但缺少 clean 原生生成链和配套 checkpoint。

所以 M0 是在锁定候选五折和可用 legacy DCE8 cache 上**新训练的受控 Next-State baseline**，复现的是 image-only next-state objective、核心 encoder/EMA/causal transition 设计与预算，不是对缺失的 native clean 数值结果的声明。

legacy DCE8 的第 8 通道是 binary ROI mask。M0/M1/M2 都不输入独立 9-D geometry、clinical、treatment 或 radiomics，但仍使用该 mask，因此本报告统一称为**ROI辅助 image-only**，而不是严格纯影像。FTV 与 mask 体素数跨访视 Spearman=0.935，FTV change 与 mask change 的三个相邻 transition Spearman=0.870/0.737/0.728；这会限制 FTV grounding 的独立生物学解释。

## 4. 方法

### 4.1 共同架构与训练

- 3-D residual CNN encoder，8 通道，base channels 16；latent dim 192；online/EMA target encoder。
- 无 clinical、treatment、geometry 或 radiomics 条件的 3 层 causal Transformer；4 heads，MLP dim 512。
- 训练 fold 使用全部 I-SPY2 train patients，并额外加入 156 名 complete-four-visit I-SPY1 做无 pCR image pretraining；I-SPY1 不进入 validation、test 或 readout。
- batch 32，最多 12 epoch，AdamW，learning rate `5e-5`，weight decay `1e-4`，EMA 0.996，step weights 2/1/0.5，SIGReg 0.09。
- M0 以 validation state loss 选 checkpoint；M1/M2 以 validation raw aggregate transition gain 选 checkpoint，并要求跨患者 latent feature std 通过门槛。pCR、test 和 radiomics loss都不参与 checkpoint 排序。

### 4.2 M0、M1、M2

| 模型 | 预测定义 | 训练损失要点 | 推理输入 |
|---|---|---|---|
| M0 | 直接预测 `z_(t+1)` | next-state LayerNorm MSE + SIGReg | 已观察 DCE8 prefix |
| M1 | `z_hat_(t+1)=z_t+delta_z_hat` | SmoothL1 delta + state reconstruction + SIGReg | 同 M0 |
| M2 | 在 M1 的 `delta_z_hat` 上接 `H_rad` | M1 + 0.05×masked radiomics SmoothL1 | 同 M0；head/radiomics 不参与 readout |

M1 先在 fold 0 validation 比较 delta-only 与 delta+state。delta-only raw gain/余弦较好，但 normalized gain=-1.39%；delta+state normalized gain=+3.61%且 state loss更低，因此按预注册的 next-state 可比性规则锁定 delta+state。详见 [m1_variant_pilot.md](m1_variant_pilot.md)。

M2 只在 fold 0 validation 比较 λ={0.05,0.1,0.25,0.5}。四者均通过 image 5% degradation 与无坍塌门槛；MAE 均在最优值 1% 内，故取最小 λ=0.05。最终 v3 selector 从 frozen validation outputs 重算 11 项 image 指标、逐项核对 history、独立重拟合 train-only transform，并沿 patient 维检查 576 个 transition×latent coordinate；独立复审未发现影响 λ 有效性的 blocker/high-risk。正式记录见 [m2_lambda_selection_v3.md](m2_lambda_selection_v3.md)。

### 4.3 Frozen pCR readout

每个决策点的 readout feature 固定为：

```text
concat(current_target_state, predicted_next_state, predicted_delta)  # 576-D
```

T0/T0–T1/T0–T2 分别只观察 T0、T0+T1、T0+T1+T2 MRI，并预测下一访状态；不读取真实未来 MRI。每折以 train-only StandardScaler + class-balanced LogisticRegression 拟合，penalty/C 和 Youden threshold 只由 validation 选择。所有 encoder/transition 均处于 eval/frozen 状态。

## 5. M0/M1/M2 pCR 主结果

下表为完整 808 人 pooled OOF test；括号为 2,000 次 patient bootstrap 的 95% percentile CI。`fold AUROC` 为五折均值±fold 间样本标准差。文中与 CSV 的 `AUPRC` 均由 scikit-learn `average_precision_score` 计算，即 average precision（AP）定义。

| 模型 | 决策点 | fold AUROC | OOF AUROC（95% CI） | OOF AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| M0 | T0 | 0.538±0.043 | 0.549（0.507–0.591） | 0.370 | 0.501 | 0.578 | 0.462 |
| M0 | T0–T1 | 0.511±0.028 | 0.504（0.463–0.545） | 0.343 | 0.472 | 0.636 | 0.386 |
| M0 | T0–T2 | 0.561±0.026 | 0.549（0.508–0.590） | 0.405 | 0.507 | 0.658 | 0.430 |
| M1 | T0 | 0.524±0.027 | 0.516（0.473–0.559） | 0.356 | 0.484 | 0.585 | 0.432 |
| M1 | T0–T1 | 0.537±0.035 | 0.539（0.498–0.580） | 0.372 | 0.540 | 0.458 | 0.582 |
| M1 | T0–T2 | 0.537±0.028 | 0.541（0.499–0.584） | 0.379 | 0.479 | 0.669 | 0.381 |
| M2 | T0 | 0.521±0.027 | 0.508（0.468–0.549） | 0.348 | 0.484 | 0.600 | 0.424 |
| M2 | T0–T1 | 0.535±0.039 | 0.529（0.485–0.571） | 0.363 | 0.532 | 0.491 | 0.553 |
| M2 | T0–T2 | 0.534±0.028 | 0.541（0.498–0.582） | 0.374 | 0.469 | 0.735 | 0.332 |

没有模型在三个决策点形成一致提升，所有 AUROC 接近 0.5，且 sensitivity/specificity 受 validation threshold 影响较大。T0–T1 的 M1 点估计高于 M0，但不能抵消 T0 与 T0–T2 的下降；M2 也没有超过 M1。

同一 808 名患者的 paired bootstrap 对比进一步确认：M1−M0 的 AUROC 95% CI 在 T0、T0–T1、T0–T2 均跨零，分别为 -0.033（-0.081, 0.017）、+0.034（-0.020, 0.088）、-0.008（-0.058, 0.041）。M2−M1 在 T0 和 T0–T1 是幅度很小但 CI 不跨零的下降：-0.008（-0.015, -0.002）与 -0.009（-0.019, -0.001）；T0–T2 为 +0.0001（-0.012, 0.012）。因此 M2 不只是“未证明改善”，在前两个决策点还呈现轻微退化。

## 6. Next-Change 与 copy-current

`normalized gain=(copy error−learned error)/copy error`；正值表示优于复制当前 latent。raw gain 使用未 LayerNorm 的 latent MSE。

| 模型 | Transition | N | normalized gain | raw gain | delta cosine | 逐患者正 gain 比例 |
|---|---|---:|---:|---:|---:|---:|
| M0 | T0→T1 | 808 | -53.90% | -125.68% | 0.292 | 19.18% |
| M0 | T1→T2 | 808 | -61.25% | -133.24% | 0.295 | 18.07% |
| M0 | T2→T3 | 808 | -71.39% | -142.00% | 0.280 | 14.98% |
| M0 | 全部 | 808 | **-62.21%** | **-133.66%** | 0.289 | 17.41% |
| M1 | T0→T1 | 808 | +7.37% | -0.93% | 0.120 | 51.61% |
| M1 | T1→T2 | 808 | +5.85% | -2.97% | 0.115 | 50.74% |
| M1 | T2→T3 | 808 | +5.35% | -4.80% | 0.076 | 46.78% |
| M1 | 全部 | 808 | **+6.24%** | **-2.84%** | 0.104 | 49.71% |
| M2 | T0→T1 | 808 | +7.36% | -0.94% | 0.119 | 51.61% |
| M2 | T1→T2 | 808 | +5.87% | -2.96% | 0.115 | 50.74% |
| M2 | T2→T3 | 808 | +5.41% | -4.73% | 0.077 | 47.28% |
| M2 | 全部 | 808 | **+6.26%** | **-2.82%** | 0.103 | 49.88% |

M1/M2 在 normalized 坐标系确实不再依赖简单 copy-current；但 raw gain 仍为负、方向余弦低且不到一半患者总体受益，因此不能表述为已准确学习真实 latent change 幅度。

## 7. Radiomics grounding

M2 原生 head 只读取 predicted image delta，未使用 pCR。下表使用每折 train-only transform 后的 standardized OOF test 值；每个单元 N=375。

| Transition | Feature | MAE | RMSE | Spearman | Pearson | R² | 方向准确率 | 预测/目标方差比 | 相对 train-mean RMSE gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0→T1 | FTV | 0.486 | 0.624 | -0.022 | -0.081 | -0.069 | 0.445 | 0.0104 | -2.61% |
| T0→T1 | Sphericity | 0.604 | 0.828 | -0.088 | -0.091 | -0.092 | 0.440 | 0.0047 | -3.40% |
| T0→T1 | LD | 0.525 | 0.970 | 0.026 | 0.034 | -0.004 | 0.403 | 0.0039 | +0.07% |
| T0→T1 | BPE | 0.858 | 1.232 | 0.006 | 0.007 | -0.065 | 0.549 | 0.0021 | -2.83% |
| T1→T2 | FTV | 0.693 | 0.872 | -0.002 | -0.030 | -0.146 | 0.603 | 0.0077 | -6.69% |
| T1→T2 | Sphericity | 0.814 | 1.181 | -0.027 | 0.005 | -0.044 | 0.485 | 0.0029 | -1.86% |
| T1→T2 | LD | 1.402 | 2.463 | 0.053 | 0.023 | -0.180 | 0.608 | 0.0010 | -8.22% |
| T1→T2 | BPE | 0.686 | 0.963 | 0.038 | 0.070 | 0.004 | 0.507 | 0.0029 | +0.45% |
| T2→T3 | FTV | 0.655 | 0.858 | 0.049 | 0.026 | -0.003 | 0.467 | 0.0060 | -0.06% |
| T2→T3 | Sphericity | 0.953 | 1.355 | -0.031 | -0.037 | -0.036 | 0.509 | 0.0024 | -1.48% |
| T2→T3 | LD | 1.441 | 2.409 | -0.024 | 0.005 | -0.128 | 0.477 | 0.0010 | -6.17% |
| T2→T3 | BPE | 0.639 | 0.914 | -0.091 | -0.110 | -0.031 | 0.469 | 0.0047 | -0.84% |

11/12 单元的预测方差小到被聚合器标记为 near-constant；唯一未标记的 T0→T1 FTV 仍只有 1.04% 的目标方差且相关为负。T1→T2 BPE 的 RMSE gain 仅 0.45%，不构成可靠可预测性。M0/M1/M2 predicted-delta 的统一 train-only Ridge probe 平均单元 R² 分别为 -0.028/-0.041/-0.041，进一步说明 M2 没有给 common image delta 增加可提取的 measurement 信息。

## 8. Shortcut resistance

所有干预复用 native frozen readout；time embedding 保持原位置。

| 模型 | 决策点/干预 | Native AUROC | 干预 AUROC | ΔAUROC（95% bootstrap CI） | 平均绝对概率变化 |
|---|---|---:|---:|---:|---:|
| M0 | T0–T1 repeated-T0 | 0.504 | 0.519 | +0.015（-0.032, 0.059） | 0.099 |
| M0 | T0–T2 repeated-T0 | 0.549 | 0.524 | -0.025（-0.071, 0.024） | 0.201 |
| M0 | T0–T2 temporal shuffle | 0.549 | 0.546 | -0.003（-0.052, 0.043） | 0.196 |
| M1 | T0–T1 repeated-T0 | 0.539 | 0.510 | -0.029（-0.080, 0.022） | 0.197 |
| M1 | T0–T2 repeated-T0 | 0.541 | 0.494 | -0.046（-0.095, 0.001） | 0.188 |
| M1 | T0–T2 temporal shuffle | 0.541 | 0.530 | -0.011（-0.055, 0.034） | 0.147 |
| M2 | T0–T1 repeated-T0 | 0.529 | 0.505 | -0.024（-0.076, 0.030） | 0.198 |
| M2 | T0–T2 repeated-T0 | 0.541 | 0.482 | **-0.058（-0.108, -0.013）** | 0.172 |
| M2 | T0–T2 temporal shuffle | 0.541 | 0.522 | -0.018（-0.065, 0.027） | 0.134 |

M1/M2 的 repeated-T0 点估计下降通常大于 M0，且 M2 T0–T2 的 CI 不含零，说明真实 follow-up 内容并非完全被忽略。但 temporal shuffle 的 CI 全部跨零，无法证明模型可靠利用了访视顺序而不是静态身份、内容集合或时间 token。

## 9. C0/C1/C2 控制组

三组都限于相同 375 名 paired OOF test 患者。C0/C1 只使用决策点前已观察到的 measurement prefix；没有未来值。

| 控制 | 推理依赖 | T0 AUROC/AUPRC | T0–T1 AUROC/AUPRC | T0–T2 AUROC/AUPRC |
|---|---|---:|---:|---:|
| C0 radiomics-only | observed measurement | 0.564/0.350 | 0.687/0.443 | **0.756/0.515** |
| C1 M2 image + radiomics | MRI + observed measurement | 0.561/0.332 | 0.541/0.310 | 0.711/0.503 |
| C2 M2 ROI辅助 image-only | MRI only（含 ROI mask channel） | 0.486/0.289 | 0.505/0.297 | 0.535/0.321 |

C0 的纵向增益说明 measurement 本身与 pCR 有关系；C1 在 T0–T2 仍较高，但推理依赖表格，不能支撑主要主张。C2 是唯一符合主要推理边界的控制，它没有接近 C0/C1。C1 低于 C0 也提示在 375 人小样本下拼接高维 image feature 可能增加估计方差，而不是提供互补信息。

## 10. M3、数据效率与停止项

M3 **按门控未运行**。M2 稳定、image AUROC 未比 M1 下降超过 0.02，validation SmoothL1 也略优于 fold-train mean；但 validation 按 feature 合并 transition 的 Spearman 仅 -0.024～0.067。单个 T1→T2 LD 点估计虽为 0.244，其预测/目标方差比只有 0.0007，属于近常数微小排序，正式 OOF 也未复现。把这种 head 变成 patient-pair relational target 会放大噪声。逐项决定见 [m3_gate.md](m3_gate.md)。

25%/50%/75% 数据效率实验未运行。原因是它属于 Priority 4 可停止项，而主五折已经显示 M2≈M1、grounding 失败；在没有先修复 target/head/ROI 重复性的情况下扩展为多比例×三模型×五折，不能合理回答“少数据时是否更有效”。因此问题 e 当前是**未知，而不是否**。

7-channel 去 ROI mask 敏感性也未运行。由于 M2 没有取得正增益，本轮优先保留完整主五折、controls 和 shortcut；代价是不能判断去除 mask 后 FTV 重复性是否下降，也不能把结果外推为严格无 ROI 辅助的 image-only 模型。

## 11. 失败实验与负面证据

- M0 原生 clean 数值复现不可执行：缺 checkpoint/cache/五折 provenance；改为明确标记的新训练受控 baseline。
- M1 delta-only 在 raw gain 与 delta cosine 上优于 delta+state，但 normalized next-state 仍劣于 copy；未进入正式五折。
- M1/M2 的 raw latent gain仍为负，说明 normalization 选择会影响“击败 copy”的结论。
- M2 λ 从 0.05 增至 0.5 没有产生有意义的 validation grounding；加权 radiomics/shared image gradient ratio从 0.113 增至 1.574，却仍接近均值预测，故未继续扩大 λ。
- M2 没有改善完整 cohort pCR；C1 fusion 的较高 paired-subset AUROC不能替代 C2。
- M3 因 grounding 门控失败；数据效率与 7-channel sensitivity 按停止规则未运行。

## 12. 泄漏与完整性审计

- 原始 fold manifest、共享数据、cache、既有 `shortcut_audit/` 和任何原始 checkpoint 均未覆盖、删除或移动。
- 五折 train/val/test patient ID 无交集；每名 808 cohort 患者恰好一次进入 test。
- radiomics transform 仅由 fold train IDs 拟合；M2 lambda 只由 fold 0 validation 选择。selector 底层会读取锁定 full-cohort pCR/raw measurement values做 contract/hash 完整性验证，但 test 值不进入候选指标或排序，也不加载 test DCE。
- pCR 只进入冻结 logistic readout；不进入 encoder、transition、M0/M1/M2 objective、radiomics head 或 checkpoint 选择。
- 正式 readout 不含 radiomics、clinical、treatment 或独立 geometry；M2 head在推理/readout中不用。
- 所有 best checkpoint 采用 `weights_only=True` 安全加载；输出 namespace 含 checkpoint/evaluator/controls hash并默认拒绝覆盖。
- 严格聚合验证 3×5 folds、808 OOF patients/model、375 paired controls、4,500 head cells；`input_issues.csv` 为 0 行。
- Paired AUROC/AUPRC bootstrap 在固定表格遍历顺序中复用一个确定性 RNG 流；完整聚合可由实现哈希、固定 seed 和输入 manifest 重现，但 CSV 单行的 seed 不应被解释为可脱离该遍历顺序独立重放的 seed。

## 13. 局限性

1. measurement 只有 FTV、sphericity、LD、BPE 四个低维特征，不代表高维 radiomics；LD/BPE 单位未明。
2. 53.59% MRI cohort 无 measurement，且 complete-case pCR、亚型、治疗与 ROI 负担不同。
3. 第 8 通道 ROI mask 与 FTV 高度同源；没有 7-channel 对照，不能声称完全排除 geometry shortcut。
4. 候选五折 manifest 没有 clean 原生 provenance；M0 是受控新训练，不是历史数值复现。
5. readout AUROC 接近随机，单折阈值和小样本 validation 会造成较大 sensitivity/specificity 波动。
6. EMA latent 的 raw scale 随训练变化；normalized gain 与 raw gain结论相反，说明 latent metric 仍需更强校准。
7. 只用一个共同训练 seed 策略，未做多 seed 稳健性；`deterministic_algorithms=false`，虽保存 seed/版本/哈希，也不保证跨硬件 bitwise 重现。
8. 未运行数据效率、M3、局部 lesion response 或更强 encoder，不能对这些扩展给出经验结论。

## 14. 对 a–j 的明确回答

| 问题 | 回答 |
|---|---|
| a. Next-Change 是否优于 Next-State？ | **对 normalized transition objective 是；对 pCR readout 不是一致地是。**M1 总 gain +6.24% 对 M0 -62.21%，但三个 pCR 决策点只有 T0–T1 的点估计提高。 |
| b. 是否更明显优于 copy-current？ | **是，但仅在 LayerNorm latent 空间。**raw gain 仍为 -2.84%，所以不能宣称全面击败 copy。 |
| c. Radiomics auxiliary 后 image representation 是否改善？ | **没有可见改善。**M2 pCR≈M1、transition≈M1、统一 post-hoc grounding probe也≈M1。 |
| d. 改善是否在不用 radiomics 推理时成立？ | 推理边界确实保持 ROI辅助 image-only，但由于 c 没有改善，答案是**不存在可确认的无表格增益**。 |
| e. 少量训练患者时是否更有效？ | **未知。**数据效率实验按预注册停止规则未运行，不能用主五折推测。 |
| f. 是否更依赖真实 follow-up 而非 repeated-T0/time token？ | **部分。**M2 T0–T2 repeated-T0 ΔAUROC=-0.058，CI不含零；但 temporal shuffle CI跨零，尚不能证明可靠编码顺序。 |
| g. 哪些 radiomics change 可可靠预测？ | **没有。**12 个单元 Spearman均接近零，11/12 近常数，仅两个单元对均值 baseline 有不足0.5%的 RMSE gain。 |
| h. 哪些 feature 提供有效监督，哪些与 geometry 重复？ | **无 feature 显示有效 patient-level监督。**FTV 与 ROI mask 强重复却仍未被可靠预测；LD同属尺寸信息，sphericity/BPE也未表现出可靠 grounding。BPE 理论上更功能性，但本轮仅 T1→T2 有0.45% RMSE gain，不足以支持。 |
| i. 是否存在 complete-case bias？ | **存在明确描述性偏差。**paired组 pCR低8.77个百分点、ROI burden更大，亚型/治疗构成也不同。 |
| j. 下一步值得加入什么？ | **暂不值得直接加 relational loss。**优先做 7-channel 去 mask、局部 lesion-response JEPA、更稳的 delta normalization/EMA坐标、功能性 target 质量核验与更强 pretrained encoder；只有 validation grounding非近常数后再考虑 M3。 |

## 15. 下一步建议

1. 先做 7-channel 严格 image-only 与 mask-only/ROI-volume control，拆分 FTV 的几何重复性。
2. 将 global pooled latent change 改为 lesion-local response tokens或局部 JEPA；BPE可考虑乳腺背景/病灶外区域的专门编码。
3. 校准 online/EMA坐标和 delta scale，尝试显式 whitened change、cosine+raw dual objective或可验证的 alpha-delta，而不是继续增加辅助权重。
4. 对 measurement target 做 scanner/site、治疗间隔与零 ROI 的质量分层；引入多 seed 前先确保每个 feature×transition 的 validation variance和相关性超过均值 baseline。
5. grounding 成功后再运行 25%/50%/75% 数据效率和 relational loss；所有扩展继续保持 pCR/test 不参与 representation 训练。

## 16. 结果索引

- 严格聚合：`../metrics/final/final_analysis_v2/`
- 九张核心图：`../figures/final/final_analysis_v2/`
- 五折 checkpoint：`../checkpoints/m0_final/`、`m1_final/`、`m2_final/`
- prediction-level 输出：`../predictions/m0_final/`、`m1_final/`、`m2_final/`、`controls/c0_c1_c2/`
- 数据审计：`../data_audit/`
- 实验计划：[EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md)
- 最终核验：[verification_report.md](verification_report.md)
