# Goal F：临床残余 phenotype state 最终报告

## 结论

最终分类：**FACTORIZATION NOT SUPPORTED**。

四个预注册 gate：A=FAIL，B=FAIL，C=FAIL，D=FAIL。本结果是两个训练种子、五个 outer folds 的 internal OOF pilot，不是外部验证。

## 十二个明确问题

1. **response performance 是否保留？** 否。z_R 相对 F0 的 static degradation floor 为 -0.03；逐 seed 结果见 response table。ΔFTV 以“同一 arm 两个 seed 均下降”定义 systematic degradation。
2. **z_P 是否 noncollapsed？** F1=是，F2=是。阈值为 mean per-dimension std ≥ 0.05、effective rank ≥ 10.0。
3. **factorization 是否降低 z_R/z_P redundancy？** 是。比较的是每个 seed 内五个 fold outer-test standardized cross-covariance RMS 的非加权均值，并要求两个 seed 均严格低于 F0；F0 control 是未分离 192-D state 的预注册前/后 96 维描述性切分，不赋予其生物学含义。
4. **residualization 前 z_P 能否解码 HR/HER2？** F1 的结果为：HER2 seed 2026 AUROC=0.528, HER2 seed 3026 AUROC=0.511, HR seed 2026 AUROC=0.548, HR seed 3026 AUROC=0.526。线性关联先在每个 static endpoint 转为 flip-invariant `0.5+|AUROC−0.5|`，再平均 T0/T1/T2；AUROC 在 0.5 任一侧偏离都可能表示可测 profile correlate，但不等同于因果 phenotype。
5. **F2 是否减少 HR/HER2 redundancy？** 否。两个 seed 的 mean endpoint flip-invariant decodability 均下降的 targets：无；此定义既不会把 0.45→0.40 错判为去冗余，也不会让 0.4/0.6 endpoint 在 raw macro 中互相抵消，并且同时要求 F2 不 collapse。
6. **z_P 是否保留纵向 image information？** F1=否，F2=否。要求 frozen weak-view cosine 达标、future predictor 在每 fold 的 T1→T2/T2→T3 上优于同一 EMA-target-projector 空间的 persistence baseline、且 z_P noncollapsed；T0→T1 因 export 未含 EMA-target T0 context 而不进入 baseline 比较。nearest-neighbor 只在同一 fold coordinate system 内跨 seed 比较，T0–T2 aggregate Jaccard 为 F1=0.199, F2=0.212。
7. **z_P 是否改善 MRI-only pCR？** 以 `[z_R,z_P] − z_R` 衡量：F1/seed 2026/T0-T1: ΔAUROC=-0.024 (95% CI -0.060, 0.012); F1/seed 3026/T0-T1: ΔAUROC=0.008 (95% CI -0.022, 0.039); F2/seed 2026/T0-T1: ΔAUROC=0.022 (95% CI -0.017, 0.059); F2/seed 3026/T0-T1: ΔAUROC=-0.007 (95% CI -0.036, 0.022); F1/seed 2026/T0-T2: ΔAUROC=-0.009 (95% CI -0.037, 0.019); F1/seed 3026/T0-T2: ΔAUROC=0.011 (95% CI -0.027, 0.047); F2/seed 2026/T0-T2: ΔAUROC=-0.001 (95% CI -0.030, 0.028); F2/seed 3026/T0-T2: ΔAUROC=-0.013 (95% CI -0.042, 0.015)。
8. **z_P 是否增加 clinical 之外的信息？** `C+z_R+z_P − C+z_R` 为：F1/seed 2026/T0-T1: ΔAUROC=-0.061 (95% CI -0.086, -0.035); F1/seed 3026/T0-T1: ΔAUROC=-0.017 (95% CI -0.038, 0.005); F2/seed 2026/T0-T1: ΔAUROC=-0.023 (95% CI -0.046, -0.000); F2/seed 3026/T0-T1: ΔAUROC=-0.035 (95% CI -0.057, -0.013); F1/seed 2026/T0-T2: ΔAUROC=-0.068 (95% CI -0.094, -0.042); F1/seed 3026/T0-T2: ΔAUROC=-0.035 (95% CI -0.062, -0.008); F2/seed 2026/T0-T2: ΔAUROC=-0.066 (95% CI -0.099, -0.034); F2/seed 3026/T0-T2: ΔAUROC=-0.034 (95% CI -0.060, -0.009)。直接的 `C+z_P − C` 为：F1/seed 2026/T0-T1: ΔAUROC=-0.075 (95% CI -0.113, -0.037); F1/seed 3026/T0-T1: ΔAUROC=-0.017 (95% CI -0.056, 0.024); F2/seed 2026/T0-T1: ΔAUROC=-0.030 (95% CI -0.068, 0.010); F2/seed 3026/T0-T1: ΔAUROC=-0.022 (95% CI -0.058, 0.014); F1/seed 2026/T0-T2: ΔAUROC=-0.027 (95% CI -0.063, 0.010); F1/seed 3026/T0-T2: ΔAUROC=-0.007 (95% CI -0.043, 0.029); F2/seed 2026/T0-T2: ΔAUROC=-0.029 (95% CI -0.068, 0.014); F2/seed 3026/T0-T2: ΔAUROC=-0.013 (95% CI -0.050, 0.023)。
9. **z_P 是否增加 clinical+FTV 之外的信息？** Gate D=FAIL；关键 `C+F+z_R+z_P − C+F+z_R` 为：F1/seed 2026/T0-T1: ΔAUROC=-0.048 (95% CI -0.098, 0.002); F1/seed 3026/T0-T1: ΔAUROC=-0.018 (95% CI -0.057, 0.019); F2/seed 2026/T0-T1: ΔAUROC=-0.033 (95% CI -0.067, 0.001); F2/seed 3026/T0-T1: ΔAUROC=-0.010 (95% CI -0.047, 0.027); F1/seed 2026/T0-T2: ΔAUROC=-0.026 (95% CI -0.071, 0.018); F1/seed 3026/T0-T2: ΔAUROC=0.002 (95% CI -0.038, 0.041); F2/seed 2026/T0-T2: ΔAUROC=-0.009 (95% CI -0.063, 0.045); F2/seed 3026/T0-T2: ΔAUROC=0.001 (95% CI -0.040, 0.043)。positive-both-seed timings=无；strong mean ≥ +0.03 timings=无。
10. **adversarial clinical residualization 有帮助还是伤害？** F2 相对 F1 的 full clinical+FTV model：F2/seed 2026/T0-T1: ΔAUROC=0.012 (95% CI -0.033, 0.059); F2/seed 3026/T0-T1: ΔAUROC=0.009 (95% CI -0.023, 0.042); F2/seed 2026/T0-T2: ΔAUROC=0.022 (95% CI -0.029, 0.071); F2/seed 3026/T0-T2: ΔAUROC=-0.004 (95% CI -0.033, 0.025)。必须与 Gate C 的去冗余结果一起解读；降低 HR/HER2 AUROC 单独不算成功。
11. **最终 state 应是 single-state 还是 factorized？** 回到单一 LOCAL response state；后续优先 patch-token/world-model 目标。
12. **这对 HR/HER2-complementary MRI information 意味着什么？** 当前目标没有证明稳定的 HR/HER2-complementary MRI 信息；这既可能是目标函数未恢复信号，也可能反映条件 MRI ceiling 较低。

## State diagnostics（outer-test、outcome-free aggregate）

| Arm   |   zP_mean_std |   zP_effective_rank |   crosscov_rms |   augmentation_cosine |   future_mse |   future_gain_over_persistence |
|:------|--------------:|--------------------:|---------------:|----------------------:|-------------:|-------------------------------:|
| F1    |         0.13  |              15.356 |          0.255 |                     1 |        0.017 |                         -0.009 |
| F2    |         0.132 |              14.875 |          0.26  |                     1 |        0.018 |                         -0.009 |

逐维 variance/std 与完整 covariance eigenspectrum 分别规范化存于 `metrics/state_dimension_diagnostics.csv` 与 `metrics/state_covariance_eigenspectra.csv`；CCA 为 ridge-regularized、supplied rows 内描述统计。没有用 t-SNE/UMAP 作主要证据。

## Response metrics

|   seed_base | arm   | state   | task   |   spearman |   rmse |    r2 |
|------------:|:------|:--------|:-------|-----------:|-------:|------:|
|        2026 | F0    | F0      | delta  |      0.34  | 22.502 | 0.081 |
|        2026 | F1    | z_R     | delta  |      0.338 | 22.514 | 0.08  |
|        2026 | F2    | z_R     | delta  |      0.339 | 22.485 | 0.081 |
|        3026 | F0    | F0      | delta  |      0.3   | 22.985 | 0.036 |
|        3026 | F1    | z_R     | delta  |      0.332 | 22.637 | 0.062 |
|        3026 | F2    | z_R     | delta  |      0.339 | 22.68  | 0.059 |
|        2026 | F0    | F0      | static |      0.531 | 28.863 | 0.099 |
|        2026 | F1    | z_R     | static |      0.519 | 28.482 | 0.116 |
|        2026 | F2    | z_R     | static |      0.499 | 28.466 | 0.12  |
|        3026 | F0    | F0      | static |      0.513 | 28.407 | 0.115 |
|        3026 | F1    | z_R     | static |      0.499 | 28.592 | 0.121 |
|        3026 | F2    | z_R     | static |      0.502 | 28.035 | 0.15  |

Static probe 严格复现 confirmed LOCAL3 contract：outer-train winsor/median-IQR FTV transform、lsqr Ridge、自然尺度 inverse；ΔFTV 使用 literal `FTV(t+1)-FTV(t)`、state difference 与 outer-train target standardization。Ridge alpha 仅由 outer-validation analysis-space MSE 选择。

## Phenotype profile probes

|   seed_base | arm   | state   | target     |   auroc | flip_invariant_decodability   |
|------------:|:------|:--------|:-----------|--------:|:------------------------------|
|        2026 | F1    | z_P     | label_her2 |   0.528 | 0.528                         |
|        2026 | F2    | z_P     | label_her2 |   0.526 | 0.526                         |
|        3026 | F1    | z_P     | label_her2 |   0.511 | 0.517                         |
|        3026 | F2    | z_P     | label_her2 |   0.518 | 0.521                         |
|        2026 | F1    | z_P     | label_hr   |   0.548 | 0.548                         |
|        2026 | F2    | z_P     | label_hr   |   0.525 | 0.530                         |
|        3026 | F1    | z_P     | label_hr   |   0.526 | 0.526                         |
|        3026 | F2    | z_P     | label_hr   |   0.528 | 0.528                         |
|        2026 | F1    | z_P     | subtype    |   0.526 | NA                            |
|        2026 | F2    | z_P     | subtype    |   0.527 | NA                            |
|        3026 | F1    | z_P     | subtype    |   0.518 | NA                            |
|        3026 | F2    | z_P     | subtype    |   0.527 | NA                            |

HR/HER2 使用 balanced linear logistic probe；subtype 使用 multiclass macro one-vs-rest AUROC。所有 C/alpha 只在 outer validation 选择。

## pCR complementarity 与 paired bootstrap

| arm   |   seed_base | population       | timing   | comparison                |   delta_auroc |   delta_auroc_ci_lower |   delta_auroc_ci_upper |   n_bootstrap |
|:------|------------:|:-----------------|:---------|:--------------------------|--------------:|-----------------------:|-----------------------:|--------------:|
| F1    |        2026 | full_808         | T0-T1    | MRI_full_vs_zR            |        -0.024 |                 -0.06  |                  0.012 |          2000 |
| F1    |        3026 | full_808         | T0-T1    | MRI_full_vs_zR            |         0.008 |                 -0.022 |                  0.039 |          2000 |
| F2    |        2026 | full_808         | T0-T1    | MRI_full_vs_zR            |         0.022 |                 -0.017 |                  0.059 |          2000 |
| F2    |        3026 | full_808         | T0-T1    | MRI_full_vs_zR            |        -0.007 |                 -0.036 |                  0.022 |          2000 |
| F1    |        2026 | full_808         | T0-T2    | MRI_full_vs_zR            |        -0.009 |                 -0.037 |                  0.019 |          2000 |
| F1    |        3026 | full_808         | T0-T2    | MRI_full_vs_zR            |         0.011 |                 -0.027 |                  0.047 |          2000 |
| F2    |        2026 | full_808         | T0-T2    | MRI_full_vs_zR            |        -0.001 |                 -0.03  |                  0.028 |          2000 |
| F2    |        3026 | full_808         | T0-T2    | MRI_full_vs_zR            |        -0.013 |                 -0.042 |                  0.015 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T1    | adversarial_F2_vs_F1_full |         0.012 |                 -0.033 |                  0.059 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T1    | adversarial_F2_vs_F1_full |         0.009 |                 -0.023 |                  0.042 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T2    | adversarial_F2_vs_F1_full |         0.022 |                 -0.029 |                  0.071 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T2    | adversarial_F2_vs_F1_full |        -0.004 |                 -0.033 |                  0.025 |          2000 |
| F1    |        2026 | ftv_complete_375 | T0-T1    | beyond_C_F_full_vs_zR     |        -0.048 |                 -0.098 |                  0.002 |          2000 |
| F1    |        3026 | ftv_complete_375 | T0-T1    | beyond_C_F_full_vs_zR     |        -0.018 |                 -0.057 |                  0.019 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T1    | beyond_C_F_full_vs_zR     |        -0.033 |                 -0.067 |                  0.001 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T1    | beyond_C_F_full_vs_zR     |        -0.01  |                 -0.047 |                  0.027 |          2000 |
| F1    |        2026 | ftv_complete_375 | T0-T2    | beyond_C_F_full_vs_zR     |        -0.026 |                 -0.071 |                  0.018 |          2000 |
| F1    |        3026 | ftv_complete_375 | T0-T2    | beyond_C_F_full_vs_zR     |         0.002 |                 -0.038 |                  0.041 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T2    | beyond_C_F_full_vs_zR     |        -0.009 |                 -0.063 |                  0.045 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T2    | beyond_C_F_full_vs_zR     |         0.001 |                 -0.04  |                  0.043 |          2000 |
| F1    |        2026 | ftv_complete_375 | T0-T1    | beyond_C_F_zP_vs_C_F      |        -0.116 |                 -0.18  |                 -0.053 |          2000 |
| F1    |        3026 | ftv_complete_375 | T0-T1    | beyond_C_F_zP_vs_C_F      |        -0.098 |                 -0.157 |                 -0.039 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T1    | beyond_C_F_zP_vs_C_F      |        -0.1   |                 -0.165 |                 -0.032 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T1    | beyond_C_F_zP_vs_C_F      |        -0.114 |                 -0.173 |                 -0.049 |          2000 |
| F1    |        2026 | ftv_complete_375 | T0-T2    | beyond_C_F_zP_vs_C_F      |        -0.12  |                 -0.18  |                 -0.06  |          2000 |
| F1    |        3026 | ftv_complete_375 | T0-T2    | beyond_C_F_zP_vs_C_F      |        -0.096 |                 -0.151 |                 -0.037 |          2000 |
| F2    |        2026 | ftv_complete_375 | T0-T2    | beyond_C_F_zP_vs_C_F      |        -0.082 |                 -0.138 |                 -0.023 |          2000 |
| F2    |        3026 | ftv_complete_375 | T0-T2    | beyond_C_F_zP_vs_C_F      |        -0.113 |                 -0.173 |                 -0.052 |          2000 |
| F1    |        2026 | full_808         | T0-T1    | beyond_C_full_vs_zR       |        -0.061 |                 -0.086 |                 -0.035 |          2000 |
| F1    |        3026 | full_808         | T0-T1    | beyond_C_full_vs_zR       |        -0.017 |                 -0.038 |                  0.005 |          2000 |
| F2    |        2026 | full_808         | T0-T1    | beyond_C_full_vs_zR       |        -0.023 |                 -0.046 |                 -0     |          2000 |
| F2    |        3026 | full_808         | T0-T1    | beyond_C_full_vs_zR       |        -0.035 |                 -0.057 |                 -0.013 |          2000 |
| F1    |        2026 | full_808         | T0-T2    | beyond_C_full_vs_zR       |        -0.068 |                 -0.094 |                 -0.042 |          2000 |
| F1    |        3026 | full_808         | T0-T2    | beyond_C_full_vs_zR       |        -0.035 |                 -0.062 |                 -0.008 |          2000 |
| F2    |        2026 | full_808         | T0-T2    | beyond_C_full_vs_zR       |        -0.066 |                 -0.099 |                 -0.034 |          2000 |
| F2    |        3026 | full_808         | T0-T2    | beyond_C_full_vs_zR       |        -0.034 |                 -0.06  |                 -0.009 |          2000 |
| F1    |        2026 | full_808         | T0-T1    | beyond_C_zP_vs_C          |        -0.075 |                 -0.113 |                 -0.037 |          2000 |
| F1    |        3026 | full_808         | T0-T1    | beyond_C_zP_vs_C          |        -0.017 |                 -0.056 |                  0.024 |          2000 |
| F2    |        2026 | full_808         | T0-T1    | beyond_C_zP_vs_C          |        -0.03  |                 -0.068 |                  0.01  |          2000 |
| F2    |        3026 | full_808         | T0-T1    | beyond_C_zP_vs_C          |        -0.022 |                 -0.058 |                  0.014 |          2000 |
| F1    |        2026 | full_808         | T0-T2    | beyond_C_zP_vs_C          |        -0.027 |                 -0.063 |                  0.01  |          2000 |
| F1    |        3026 | full_808         | T0-T2    | beyond_C_zP_vs_C          |        -0.007 |                 -0.043 |                  0.029 |          2000 |
| F2    |        2026 | full_808         | T0-T2    | beyond_C_zP_vs_C          |        -0.029 |                 -0.068 |                  0.014 |          2000 |
| F2    |        3026 | full_808         | T0-T2    | beyond_C_zP_vs_C          |        -0.013 |                 -0.05  |                  0.023 |          2000 |

每个比较使用 2,000 次 paired patient bootstrap，并在 outer-fold × pCR outcome 精确 strata 内重采样；AUROC/AUPRC 为 augmented−baseline，Brier improvement 为 baseline−augmented。patient-level OOF probabilities 与 bootstrap draws 仅写入 gitignored `predictions/`。

## 模型与 cohort contract

- F0：既有 LOCAL3 192-D response state；F1/F2：96-D z_R + 96-D z_P，总维数固定 192。
- Confirmed LOCAL3 的 z_R transition 保持 canonical image-only；新增 z_P future predictor 保留 causal treatment/HR/HER2/MP condition。phenotype query 本身不接收 clinical/treatment，treatment 也不作 adversarial removal。
- Representation training 明确禁止读取 pCR；本阶段只接受完整且 hash/provenance 绑定的 frozen exports 后才加载 pCR。
- Clinical C：HR、HER2、MP、screening age、treatment arm；类别词表与缺失值处理只在 outer train 拟合。
- FTV complete cohort 固定 n=375；全 cohort 固定 n=808。时点为 T0、T0–T1、T0–T2。
- F3 在本次 primary run 中关闭；它仅是可选 downstream control，不是 World Model arm，且不得参与主分类。

## Goal B ceiling 的条件解释

本报告不伪造尚未接入的 Goal B 结果。若 supervised conditional ceiling 强而 Goal F 失败，较合理的解释是 MRI signal 存在但 outcome-free residualization objective 未恢复；若 Goal B 也失败，有限 conditional MRI ceiling 更可信；若二者均成功，则 outcome-supervised ceiling 与 outcome-free recovery 形成最强相互支持。

## 局限

1. 两个 seed 只够 primary pilot，不能替代五 seed confirmation。
2. Internal OOF 与 paired bootstrap 不解决 dataset shift、label noise 或 external transportability。
3. F0 前/后 96-D 切分仅是 redundancy control；不能解释为天然 response/phenotype decomposition。
4. Adversarial independence 只针对线性 HR/HER2 adversary，不证明统计独立或去除所有 clinical correlate。
5. Gate D 使用 point ΔAUROC 的方向一致性；CI 同时报告但不是预注册 pass 条件。
