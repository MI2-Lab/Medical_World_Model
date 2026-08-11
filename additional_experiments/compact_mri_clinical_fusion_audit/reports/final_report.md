# Compact MRI–Clinical Fusion Audit — 最终报告

> 诊断性/探索性两种子证据。正式运行包含每个配对 cell 2,000 次 patient-level、outer-fold-stratified bootstrap。MRI seed/arm 是同一批患者上的敏感性 cell，不是独立患者样本。

## 执行结论

预注册分类为 **D. MIXED / UNSTABLE**。

证据同时支持两件事：

1. **原始高维 fusion 存在明显的有限样本过拟合。** Goal 2 的 `M192` 只在 T0 是 192-D；其真实 timing-prefix 宽度是 192/384/576/768。在 `ftv_complete_375` 中，raw `C+F+M` 的平均 train−test AUROC gap 从 T0 的 0.277 增至 T3 的 0.321；validation-selected compact PCA 将其降至 0.163–0.181。
2. **但减小维度不足以稳定揭示 clinical- 或 FTV-complementary pCR signal。** Headline PCA `C+Mk−C` 在 `full_808` 只有 T2 的四 cell 平均为 +0.004，且 0/4 CI 排除 0；在 `ftv_complete_375` 所有 timing 都为负。`C+F+Mk−(C+F)` 仅 T1 平均 +0.005，3/4 point estimates 为正，但 0/4 CI 排除 0，AUPRC 平均反而 -0.009。严格 OOF late fusion 未改善临床基线。

本实验因此不支持 A（PCA compact 稳定揭示互补信号），也不支持 B（仅 late fusion 有效）。不将结论归为干净的 C，是因为预设的 fixed R32 在 `full_808` 呈现了方向一致的改善：T1 和 T2 的四个 seed×arm cell 全部为正，平均 ΔAUROC 分别为 +0.021 和 +0.027，T2 同时有平均 ΔAUPRC +0.020 和 ΔBrier -0.024。但这一信号没有被 PCA 或 late fusion 复现，在 `ftv_complete_375` 中也反向，因而属于 method/population-dependent sensitivity，不足以升级为稳定的 complementary-signal 结论。

## 冻结范围与 estimand

| Population | n | pCR+ | 用途 |
|---|---:|---:|---|
| `full_808` | 808 | 275 (34.0%) | `C` vs `C+M`；HR/HER2/subtype probes |
| `ftv_complete_375` | 375 | 110 (29.3%) | 完全 matched C/F/M 比较；primary beyond-FTV estimand |

两个 population 的 absolute metrics 回答不同 estimand，不被解释为 paired effect。所有输入严格复用 Goal 2 的 patient IDs、folds、pCR labels、`C2_full_with_treatment`、FTV prefix、LOCAL0/LOCAL3 states 与 [information timing contract](../information_timing_contract.csv)。T3 始终标记为 late/pre-surgery，不与 early/mid evidence 混合。

本轮没有重新训练 encoder/JEPA，没有修改 LOCAL/C1B，没有新 grounding target、attention、foundation model 或 deep pCR classifier。完整预注册边界见 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md)。

## Compact representation 与 explained variance

为了直接审计 Goal 2 的 prefix fusion，本报告中 `Mk` 表示将当前可用的完整 MRI timing prefix 压缩到 k 个总维度，而非每个 visit 保留 k 维。PCA 仅在 population×seed×arm×outer-fold×timing 的 outer train 上 fit，validation/test 仅 transform。

| Timing | Raw width | PCA8 mean variance | PCA16 | PCA32 | PCA64 |
|---|---:|---:|---:|---:|---:|
| T0 | 192 | 95.8–96.1% | 98.4–98.5% | 99.5–99.5% | 99.9% |
| T1 | 384 | 91.7–92.3% | 96.3–96.7% | 98.6–98.8% | 99.5–99.7% |
| T2 | 576 | 87.7–88.4% | 94.5–95.0% | 97.7–98.1% | 99.2–99.4% |
| T3 | 768 | 83.5–84.4% | 92.6–93.3% | 96.9–97.3% | 98.8–99.1% |

这表明 state 方差高度集中，但高 explained variance 不等于保留了 pCR-complementary 或 phenotype-related 低方差方向。完整 fold-level 记录见 [Table 1](../metrics/table1_pca_dimension_explained_variance.csv)、[PCA component ledger](../metrics/pca_component_explained_variance.csv) 与 [PCA figure](../figures/pca_explained_variance.png)。

### 最佳 compact dimension

没有一个可跨 model family 声称的单一最佳 k。在非-late primary/secondary PCA families 的 560 个 fold-level selections 中，64/8/32/16 维分别被选中 156/139/137/128 次（27.9%/24.8%/24.5%/22.9%），没有压倒性 mode。

- MRI-only `M` 在 `full_808` 和 `ftv_complete_375` 中都以 64 维为 mode，分别 34/80 和 37/80 folds。
- `full_808` 的 `C+M` 在 32 与 64 维各 27/80，并列。
- `ftv_complete_375` 的 `C+M` 以 8 维为 mode（27/80）；`C+F+M` 更明显倾向 8 维（36/80，45%）。

因此，如果“MRI 最佳 dimension”严格指 MRI-only validation selection，答案是 **64-D modal，但不稳定**；如果指临床/FTV incremental fusion，低容量 8-D 更常被选中。详细分布见 [selected dimensions by fold](../metrics/selected_dimensions_by_fold.csv) 和 [selection-frequency table](../metrics/dimension_selection_frequency.csv)。

## 192-D-per-visit raw fusion 是否过拟合？

**是，而且 T1–T3 非常明显。** 下表是 seed×arm×fold 上的平均 train/test AUROC；它与跨 fold pooled OOF AUROC 不同，用于诊断 generalization gap。

| Population/model | Timing | Raw train/test (gap) | Selected compact train/test (gap) |
|---|---|---|---|
| `full_808`, C+M | T0 | 0.839 / 0.683 (0.157) | 0.807 / 0.698 (0.110) |
|  | T1 | 0.868 / 0.688 (0.180) | 0.800 / 0.710 (0.090) |
|  | T2 | 0.886 / 0.692 (0.194) | 0.810 / 0.712 (0.099) |
|  | T3 late | 0.923 / 0.676 (0.247) | 0.818 / 0.716 (0.103) |
| `ftv_complete_375`, C+F+M | T0 | 0.903 / 0.625 (0.277) | 0.841 / 0.678 (0.163) |
|  | T1 | 0.949 / 0.662 (0.287) | 0.869 / 0.713 (0.156) |
|  | T2 | 0.980 / 0.665 (0.315) | 0.891 / 0.718 (0.172) |
|  | T3 late | 0.978 / 0.657 (0.321) | 0.896 / 0.715 (0.181) |

Goal 2 已经对完整 concatenated train matrix 做了 train-only column standardization，所以本轮发现主要指向 dimensionality/noise/regularization，而非“没有做基本 scale balancing”。全部诊断见 [overfitting table](../metrics/overfitting_diagnostics.csv) 和 [raw-vs-compact figure](../figures/raw_vs_compact.png)。

## MRI-only：raw 与 compact

AUROC 为四个 seed×arm cells 的平均（最小–最大）。

| Population | Timing | Raw M | Selected compact M | Compact − raw |
|---|---|---:|---:|---:|
| `full_808` | T0 | 0.520 (0.506–0.538) | 0.524 (0.513–0.546) | +0.004 |
|  | T1 | 0.535 (0.526–0.543) | 0.537 (0.526–0.546) | +0.002 |
|  | T2 | 0.523 (0.516–0.531) | 0.539 (0.528–0.549) | +0.016 |
|  | T3 late | 0.546 (0.542–0.548) | 0.548 (0.519–0.573) | +0.003 |
| `ftv_complete_375` | T0 | 0.512 (0.477–0.549) | 0.487 (0.467–0.517) | -0.024 |
|  | T1 | 0.521 (0.509–0.539) | 0.511 (0.487–0.532) | -0.010 |
|  | T2 | 0.530 (0.479–0.567) | 0.512 (0.483–0.533) | -0.018 |
|  | T3 late | 0.549 (0.526–0.577) | 0.539 (0.501–0.576) | -0.010 |

Compact MRI-only 因此只在 `full_808`、尤其 T2 有小幅改善；在 FTV-complete 小样本中反而更差。Brier 普遍下降，但这主要反映概率收缩/校准，不能替代 discrimination evidence。见 [Table 2](../metrics/table2_mri_only_raw_vs_compact.csv) 与 [AUROC-by-dimension figure](../figures/auroc_by_dimensionality.png)。

## Primary incremental effects

正的 ΔAUROC/ΔAUPRC 有利于 augmented model；负的 ΔBrier 有利。“CI envelope”是四个分别计算的 seed×arm cell CI 中最小 lower 到最大 upper，**不是 pooled CI**；这些 cell 复用同一批患者，彼此不是独立患者样本。

| Comparison | Population | Timing | Mean ΔAUROC (cell range) | Cell-CI envelope | Positive points | Entirely-positive CIs | Mean ΔAUPRC | Mean ΔBrier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Δ1 C+Mk−C | `full_808` | T0 | -0.013 (-0.017 to -0.006) | -0.047 to +0.021 | 0/4 | 0/4 | -0.030 | -0.004 |
|  |  | T1 | -0.012 (-0.013 to -0.008) | -0.042 to +0.017 | 0/4 | 0/4 | -0.025 | -0.002 |
|  |  | T2 | +0.004 (-0.004 to +0.010) | -0.035 to +0.036 | 3/4 | 0/4 | +0.001 | -0.001 |
|  |  | T3 late | -0.006 (-0.023 to +0.013) | -0.060 to +0.043 | 2/4 | 0/4 | -0.012 | -0.002 |
| Δ1 C+Mk−C | `ftv_complete_375` | T0 | -0.055 (-0.083 to -0.041) | -0.140 to +0.011 | 0/4 | 0/4 | -0.066 | -0.015 |
|  |  | T1 | -0.055 (-0.068 to -0.039) | -0.132 to +0.009 | 0/4 | 0/4 | -0.083 | -0.014 |
|  |  | T2 | -0.067 (-0.085 to -0.049) | -0.150 to +0.005 | 0/4 | 0/4 | -0.117 | -0.006 |
|  |  | T3 late | -0.072 (-0.091 to -0.063) | -0.153 to -0.009 | 0/4 | 0/4 | -0.100 | -0.009 |
| Δ2 C+F+Mk−C+F | `ftv_complete_375` | T0 | -0.046 (-0.068 to -0.032) | -0.125 to +0.017 | 0/4 | 0/4 | -0.068 | -0.005 |
|  |  | T1 | +0.005 (-0.008 to +0.016) | -0.051 to +0.056 | 3/4 | 0/4 | -0.009 | -0.003 |
|  |  | T2 | -0.042 (-0.080 to -0.017) | -0.130 to +0.014 | 0/4 | 0/4 | -0.067 | +0.028 |
|  |  | T3 late | -0.025 (-0.026 to -0.023) | -0.075 to +0.030 | 0/4 | 0/4 | -0.044 | +0.015 |

Headline validation-selected PCA 没有满足 A 的稳定性条件。预设 fixed-dimension curve 中，PCA8 的 `C+F+M−(C+F)` 在 T1 四 cell 全部为正，平均 +0.009；但这是查看所有 test dimension 后的 sensitivity，不能取代预先规定的 validation-selected headline。详情见 [Table 3](../metrics/table3_c_vs_c_plus_m.csv)、[Table 4](../metrics/table4_cf_vs_cf_plus_m.csv)、[dimension-delta figure](../figures/delta_auroc_vs_dimension.png) 和 [beyond-FTV figure](../figures/beyond_ftv_delta_auroc.png)。

## Compact 是否只是改善了 raw fusion？

是，但改善幅度通常只是把明显失败的 raw joint model 拉回接近 baseline：

- `ftv_complete_375` 的 compact-vs-raw `C+F+M` 平均 ΔAUROC 为 T0 +0.042、T1 +0.051、T2 +0.038、T3 +0.045。T1/T3 的四 cell point estimates 全部为正，每个 timing 各 2/4 CIs 排除 0。
- 但与 `C+F` 比，同一 compact model 的平均 ΔAUROC 为 -0.046/+0.005/-0.042/-0.025（T0/T1/T2/T3）。

因此“降维缓解 raw overfit”为真，但“降维揭示了稳定临床互补信号”不成立。见 [raw-vs-compact paired effects](../metrics/table5_raw_vs_compact_paired_effects.csv)。

## FTV residualization

FTV→compact-MRI 线性映射在每个 outer train 内 fit，然后冻结 transform validation/test。

| Timing | Residual MRI-only AUROC mean | C+F+residual Mk AUROC mean | C+F AUROC | Mean ΔAUROC joint−C+F |
|---|---:|---:|---:|---:|
| T0 | 0.494 | 0.640 | 0.688 | -0.048 |
| T1 | 0.498 | 0.693 | 0.697 | -0.004 |
| T2 | 0.509 | 0.696 | 0.726 | -0.030 |
| T3 late | 0.509 | 0.695 | 0.715 | -0.021 |

Residual MRI-only 基本为 chance，且 residual joint 没有一个 timing 在四 cell 上统一为正。这不支持“去除 FTV-associated component 后仍有可利用 pCR information”。见 [residual metrics](../metrics/residualized_compact_metrics.csv) 与 [beyond-FTV figure](../figures/beyond_ftv_delta_auroc.png)。

## Strict OOF late fusion

Meta-model 的两个输入是 `[clinical logit, MRI logit]`。每个 outer-train patient 都由一个没有见过该患者的 inner-fold clinical encoder/PCA/scaler/base model 预测一次；不使用 outer-train in-sample logits。

- `LateFusion(C,Mk)−C` 在 `full_808` 的平均 ΔAUROC 为 -0.016/-0.009/-0.017/-0.017；在 `ftv_complete_375` 为 -0.073/-0.036/-0.063/-0.049。
- `LateFusion(C+F,Mk)−(C+F)` 为 -0.045/-0.029/-0.052/-0.021。
- 只有 `full_808` T1 的某一 cell 为正且 CI 排除 0，同 timing 其他 cells 不复现。

因此不支持 B。见 [Table 6](../metrics/table6_late_fusion.csv)、[late diagnostics](../metrics/late_fusion_diagnostics.csv) 与 [late-vs-feature-fusion figure](../figures/late_vs_early_fusion.png)。

## Random projection control

R16/R32 不读取患者、标签或 test，只由固定 seed 260812 和 input/output dimension 生成。它产生了本轮最重要的 mixed sensitivity：

- `full_808` 中 R32 `C+M−C` 的平均 ΔAUROC 为 T0 +0.013、T1 +0.021、T2 +0.027、T3 +0.036。T1/T2/T3 四 cell point estimates 全部为正；排除 0 的 cell CIs 为 2/4、2/4、4/4。T2 还有平均 ΔAUPRC +0.020 与 ΔBrier -0.024。
- 同一 R32 `C+M−C` 在 `ftv_complete_375` 四个 timing 全部为负（-0.041/-0.065/-0.032/-0.049）。
- R16 `C+F+M−(C+F)` 在 T1 四 cell 全部为正，平均 +0.015，但 0/4 CIs 排除 0；R32 beyond-FTV 反而为负。

这不支持“PCA 明确删除了 noise”。更合理的诊断是：部分 weak signal 可能分散在非主方差方向，PCA 可能丢弃它；而一个 fixed random subspace 在较大样本 estimand 中偶然/真实保留了更多可用信号。由于只有一个 projection seed，且在 FTV estimand 不复现，必须重复 projection seeds 后才能作生物学结论。见 [random-projection sensitivity](../metrics/random_projection_sensitivity.csv)。

## HR/HER2/profile decodability

下表是四个 seed×arm cells 的平均 AUROC。

| Target | Timing | Raw | PCA16 | PCA32 |
|---|---|---:|---:|---:|
| HR | T0 / T1 / T2 / T3 | .523 / .509 / .532 / .534 | .515 / .511 / **.543** / .528 | .527 / .511 / .527 / .512 |
| HER2 | T0 / T1 / T2 / T3 | .519 / .556 / .557 / .563 | **.537** / .550 / **.571** / **.571** | .527 / .547 / .557 / .562 |
| 4-class subtype | T0 / T1 / T2 / T3 | .515 / .532 / .540 / .544 | .518 / **.534** / **.551** / **.547** | .508 / .521 / .543 / .544 |

PCA16 在 T2 对 HR/HER2/subtype 都有小幅改善，但 PCA32 并没有稳定超过 raw；所有最佳平均仍低于 0.60。因此 compact space 有“少量 phenotype-irrelevant dimensions 被去除”的迹象，但没有证明强 phenotype information 被原始高维 noise 完全遮蔽。见 [Table 7](../metrics/table7_profile_decodability.csv) 和 [profile figure](../figures/profile_decodability.png)。

## 对 10 个必答问题的直接回答

1. **192-D fusion 是否存在明显 overfitting？** 是。更准确地说，T0 为 192-D，T1–T3 是 384/576/768-D prefix。Raw joint train−test gap 明显大于 compact，并随 timing 增大。
2. **MRI 最佳 compact dimension？** 没有全局唯一 k。MRI-only validation mode 是 64-D；但 incremental/beyond-FTV fusion 更常选 8-D。分布分散，不应根据 test 曲线宣称单一“最佳”。
3. **Compact MRI-only 是否优于 raw MRI？** `full_808` 小幅优于，尤其 T2 +0.016；`ftv_complete_375` 所有 timing 均更差。结论不稳定。
4. **C+compact MRI 是否超过 C？** 没有稳定支持。`full_808` T2 平均 +0.004，但 0/4 CIs 排除 0；FTV population 全为负。
5. **C+F+compact MRI 是否超过 C+F？** 没有稳定支持。T1 有弱趋势（+0.005，3/4 正，0/4 CI 排除 0），其他 timing 为负。
6. **Residualized compact MRI 是否仍有 pCR signal？** 没有。MRI-only 约 0.49–0.51 AUROC，joint 不超过 `C+F`。
7. **Late fusion 是否有效？** 没有。四个 headline late-fusion deltas 的稳定条件均未满足。
8. **HR/HER2 signal 是否在 compact space 更明显？** PCA16 在部分 timing，尤其 T2，更明显；PCA32 不稳定，且整体仍弱。
9. **Goal 2 negative result 是否主要由 fusion 导致？** 不是单一主因。高维 fusion 确实放大了失败，但 PCA/late/residual 没有恢复稳定的 incremental value。R32 sensitivity 说明 fusion/regularization 仍是贡献因素，但 phenotype-poor representation 仍是核心限制。
10. **下一步应修改 fusion / representation / encoder 中哪个？** 如果必须选一个，选 **encoder**，目标是产生更强的 phenotype/kinetics-aware representation；后续 readout 保留低容量 compact/RP control。证据不支持继续只堆叠 fusion 工程。

## 局限与下一步

- 只有两个 MRI seed bases 和两个 LOCAL arms；结论仍是 diagnostic/exploratory。
- PCA dimension/C/meta-C 由 outer validation 选择，bootstrap CI 是在已选定 pipeline 下的 conditional uncertainty，不替代新 cohort replication。
- Fixed-dimension test curves 是预设 sensitivity，不能用于事后改写 headline k。
- R32 只有一个 fixed projection matrix seed。下一步应先做多 projection-seed replication，确认 `full_808` T1/T2 signal 是否稳定，但不把其当作新的 patient sample。
- `full_808` 要求四 visit 完整，`ftv_complete_375` 还条件于四个 FTV 可用；不代表所有临床患者。

建议的实验顺序是：

1. 复现 R32 多 projection seeds，并将 projection seed 作为 model sensitivity，不作为患者样本；
2. 保留当前完全冻结的 clinical/FTV/fold audit，只更换 encoder-derived representation；
3. 优先强化 lesion morphology、heterogeneity、enhancement kinetics 与 multi-scale spatial context，且避免 pCR leakage；
4. 新 representation 同时用 raw、PCA/RP compact 和 strict late fusion 读出，以区分 encoder gain 与 fusion gain。

## 产物、复现与完整性

| Item | Value |
|---|---|
| Branch | `feature/compact-mri-clinical-fusion-audit` |
| Parent commit | `064e0596348f0972decc39774336580f58e8da61` |
| Audit implementation commit | `efb1749a9c475917050ab645a5130e8babb4cdd6` (`Add compact MRI clinical fusion audit`) |
| GitHub push status | **SUCCESS** — `origin/feature/compact-mri-clinical-fusion-audit` 已创建并包含上述 implementation commit；无 force push |
| LOCAL cells | 2 seeds × 2 arms × 5 outer folds = 20 |
| Formal pCR prediction rows | 566,416 |
| Profile prediction rows | 116,352 |
| Strict late-fusion inner-OOF rows | 320,832 |
| Bootstrap comparison cells | 336 |
| Bootstrap draws | 2,000 per cell |
| Raw Goal 2 regression | PASS; AUROC/AUPRC/Brier exact within `1e-12` |

必需表格：

1. [PCA dimension / explained variance](../metrics/table1_pca_dimension_explained_variance.csv)
2. [MRI-only raw vs compact](../metrics/table2_mri_only_raw_vs_compact.csv)
3. [C vs C+M](../metrics/table3_c_vs_c_plus_m.csv)
4. [C+F vs C+F+M](../metrics/table4_cf_vs_cf_plus_m.csv)
5. [Raw vs compact paired effects](../metrics/table5_raw_vs_compact_paired_effects.csv)
6. [Late fusion](../metrics/table6_late_fusion.csv)
7. [HR/HER2 decodability](../metrics/table7_profile_decodability.csv)
8. [Bootstrap CI](../metrics/table8_bootstrap_ci.csv)

附加 reproducibility artifacts：[run summary](../metrics/run_summary.json)、[paired effects](../metrics/paired_effects.csv)、[Goal 2 raw regression check](../metrics/goal2_raw_regression_check.csv)、[random projection ledger](../metrics/random_projection_ledger.csv)、[PCA artifact manifest](../metrics/pca_artifact_manifest.csv)。公开报告/表格/图只读取 aggregate metrics；patient IDs、OOF predictions、PCA parameters 与 bootstrap draws 保留在 gitignored private artifacts 中。
