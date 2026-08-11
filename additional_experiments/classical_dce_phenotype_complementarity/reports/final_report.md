# Classical DCE Phenotype Complementarity Baseline：最终报告

## 结论先行

主分析是 `primary_stratified_384` / `clinical_radiomics_complete_384` / `complete_case` / logistic regression，共 **n=384（pCR+=113）**。按预注册判据作严格、不强制的科学映射，结果为：

> **A. CLASSICAL PHENOTYPE COMPLEMENTARITY SUPPORTED**
>
> 传统 DCE phenotype 在 FTV 之外提供稳定且 residualization 后仍存在的 pCR 信息。

该映射只由 primary radiomics 的预注册规则产生；current MRI reference、SVM 和 family ablation 均不参与类别选择。只有完整满足定义才赋 A/B/C/D；若残差信号等证据与某类前提冲突，则报告 `MIXED/INCONCLUSIVE`，而不强塞进“否则 D”或错误声称 FTV redundancy。T3 在全文与图中均标为 **late/pre-surgery**，不能与较早、可行动的 timing 等量解释。

| 预注册判据 | 观测 | 是否满足 |
| --- | --- | --- |
| C+FULL−C+F 的 ΔAUROC CI>0：≥2 timings，且至少一个 T0–T2 | T1、T2 | 是 |
| 是否存在任一孤立 C+FULL−C+F ΔAUROC CI>0（B/C/D 均要求无） | T1、T2 | 是 |
| C+F+N_res−C+F 的 ΔAUROC CI>0：≥1 timing | T1、T2 | 是 |
| N standalone AUROC≥0.60：≥1 timing | 最佳描述值 0.750 @ longitudinal/T3 (late/pre-surgery) | 是 |
| N→HR/HER2/subtype AUROC≥0.60：≥1 timing | N 最佳描述值 0.610 (N→HER2, static/T2) | 是 |

## 12 个必答问题

### 1. 实际有哪些 radiomics features？

工作簿 inventory 识别出 16 个独立 imaging measurement：FTV、LD、SPH、BPE 各覆盖 T0–T3；另有 12 个相对 T0 的派生 percent-change 列，它们不算新的独立观测。

| Family | 真实 measurement | 独立绝对值列数 | 访视 |
| --- | --- | --- | --- |
| FTV | functional tumor volume（F） | 4 | T0, T1, T2, T3 |
| LD | longest diameter（D） | 4 | T0, T1, T2, T3 |
| SPH | sphericity / shape（S） | 4 | T0, T1, T2, T3 |
| BPE | contralateral 5-slice mean enhancement（B） | 4 | T0, T1, T2, T3 |

这里的 “radiomics” 是低维、纵向 DCE measurement，不是高维 texture feature。完整字段、单位、coverage 与 leakage concern 见 [feature inventory CSV](../features/radiomics_feature_inventory.csv) 和 [可读 inventory](radiomics_feature_inventory.md)。

### 2. 哪些是 FTV，哪些属于 non-FTV phenotype？

F=FTV；NONFTV=N=D+S+B（LD、SPH、BPE）；FULL=F+NONFTV。Patient ID 只用于 join/split，绝不进入 predictor。 原始工作簿另有 percent-change 派生列，但 pipeline 从 timing-safe absolute observations 按预注册公式重建 change，避免重复计数。

### 3. non-FTV 是否单独预测 pCR？

N standalone 的最高**描述性** OOF AUROC 为 **0.750**（longitudinal/T3 (late/pre-surgery)，AUPRC=0.546）。按固定 AUROC≥0.60 综合阈值，答案为 **是**。这个“最高值”只用于描述，未用于选择 feature、timing 或 primary model。

### 4. 是否增加 Clinical-only performance？

**至少一个预注册 cell 有明确正增量**。ΔAUROC 范围 -0.012 至 0.077；95% CI 全高于 0 的 cell 为 static/T2、static/T3 (late/pre-surgery)。 因而应结合所有 timing 的 effect size/CI 阅读，不能只挑最大的 test cell；完整 ΔAUROC、ΔAUPRC 与 Brier improvement 见 incremental-effects 表。

### 5. 是否增加 Clinical+FTV performance？

ΔAUROC 范围 0.002 至 0.060；95% CI 全高于 0 的 cell 为 longitudinal/T1、static/T2。 预注册的“稳定增量”判据 **满足**；判据要求至少两个 distinct timings 的 paired ΔAUROC 95% CI 全高于 0，且至少一个来自 T0–T2。

### 6. residualized non-FTV 是否仍有 signal？

N_res 最高描述性 AUROC 为 **0.685**（longitudinal/T3 (late/pre-surgery)）。ΔAUROC 范围 0.006 至 0.066；95% CI 全高于 0 的 cell 为 longitudinal/T1、static/T2。 因而 residual-signal 判据 **满足**。NONFTV→FTV redundancy 的最高 R² 为 0.571（longitudinal/T2，Spearman=0.763）。Residualizer 与 redundancy regression 均只在 outer train 拟合。

### 7. 哪类 feature 最有价值？

按所有预注册 view/timing cell 的配对 AUROC 差均值，D 排名最高；这是 family 定位描述，不是根据 test 表现重新选择主模型。

| Family | mean ΔAUROC | min | max |
| --- | --- | --- | --- |
| D | 0.035 | 0.010 | 0.084 |
| S | 0.009 | -0.007 | 0.045 |
| B | 0.001 | -0.013 | 0.023 |

### 8. HR/HER2 是否可从传统 DCE phenotype 预测？

全部传统 feature sets 中的最佳描述性 probe 是 **FULL→HER2**，AUROC=0.643、AUPRC=0.341、balanced accuracy=0.581（longitudinal/T3 (late/pre-surgery)）。用于 C 类判定的 NONFTV-only 最佳值为 **N→HER2 AUROC=0.610**（static/T2）；按固定 N-only AUROC≥0.60 规则，profile signal **存在**。FULL 仍单独作描述，但 FULL-only crossing 不归因于 non-FTV，因为它可能由 FTV 驱动。Probe 是 cohort-level correlate，不是因果机制或可直接临床部署的 biomarker。

### 9. LR 和 SVM 结论是否一致？

C+FULL−C+F 的跨 cell 平均差：LR 0.029，RBF-SVM 0.005，因此增量方向一致。逐模型 SVM−LR AUROC 的中位数为 0.006；SVM 是 secondary sensitivity，不改变 LR primary classification。

| 模型 | mean C+FULL−C+F AUROC |
| --- | --- |
| Logistic regression | 0.029 |
| RBF SVM | 0.005 |

### 10. 当前 MRI latent 是否至少达到传统 radiomics 水平？

不同 matched cell 结论混合。

MRI reference 细节：可配对 MRI−traditional AUROC 共 25 个 cell，范围 -0.171 至 0.104；traditional N 高于 M 的 matched cell 为 7/25，最大三个差距为 pCR/longitudinal/T2 (-0.171)、pCR/longitudinal/T3 (-0.170)、pCR/longitudinal/T1 (-0.081)；证据：mri_reference_traditional_pcr_comparison.csv 的 N-vs-M direct matched descriptive comparison；mri_reference_traditional_profile_comparison.csv 的 N-vs-M direct matched descriptive comparison。 它是 supplementary sensitivity，绝不改变 384 人 primary radiomics classification。

这里的等级判断只使用同一 375 人上的 **N vs M 描述性对照**。Frozen MRI audit 的 C/F preprocessing 与 prediction head 不同于本实验：旧 F 主要是 log1p absolute prefix，新 F 包含 outer-train winsorized absolute/delta/relative features，clinical encoding 也不同。因此 `C+N vs C+M` 与 `C+FULL vs C+F+M` 有 baseline confounding，只展示在 direct comparison 表中，不能解释为 causal 或 paired incremental effect。

### 11. MRI 里是否存在 World Model 还没学到的 phenotype？

A 类结果支持在本 cohort 中传统 DCE 存在 FTV 外、对 pCR 有增量的 phenotype；同一 375 人的 N-vs-M 对照也确实存在，但方向随 target/timing 改变。traditional N 高于 M 的 matched cell 为 7/25，最大三个差距为 pCR/longitudinal/T2 (-0.171)、pCR/longitudinal/T3 (-0.170)、pCR/longitudinal/T1 (-0.081)。这些 traditional>N cell 支持 target/timing-specific representation gap 作为候选解释，却不足以宣称 World Model 全局遗漏，更不是因果证据。

### 12. 对下一版 World Model 最直接的建议是什么？

加入 phenotype-aware state/auxiliary targets（显式 FTV、LD、shape、BPE），并用 FTV-residual objective 检验 latent 是否学到非体积信息。

## 方法、matching 与 leakage control

- Primary estimand 是 strict matched complete-case；每个 paired comparison 在相同 view/timing/fold 使用完全相同患者。Manifest 的 primary-384 子集前 12 行如下，完整 cross-protocol 表见链接。

| protocol | population | scenario | comparison | view | timing | n | pCR_positive | missingness_exclusions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+F+N_res_vs_C+F | longitudinal | T0 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+FULL_vs_C+F | longitudinal | T0 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+N_vs_C | longitudinal | T0 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | all_primary_model_families | longitudinal | T0 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+F+N_res_vs_C+F | longitudinal | T1 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+FULL_vs_C+F | longitudinal | T1 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+N_vs_C | longitudinal | T1 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | all_primary_model_families | longitudinal | T1 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+F+N_res_vs_C+F | longitudinal | T2 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+FULL_vs_C+F | longitudinal | T2 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | C+N_vs_C | longitudinal | T2 | 384 | 113 | 0 |
| primary_stratified_384 | clinical_radiomics_complete_384 | complete_case | all_primary_model_families | longitudinal | T2 | 384 | 113 | 0 |

- Static `Tk` 只读当前 `Tk`；longitudinal `Tk` 只读 T0…Tk，并仅从已观察 prefix 构建 absolute/relative change。[机器可读 timing contract](../information_timing_contract.csv)明确 T3 为 late/pre-surgery。
- Log transform、1%/99% winsorization、median/missingness indicator、categorical vocabulary、scaling、FTV residualization 与 redundancy regression 均只在 outer train 拟合；validation 只选超参数/threshold；outer test 不参与 selection。
- Primary model 是 L2 logistic regression；RBF SVM 只作 sensitivity。Feature families、change formula 和模型空间均在 outcome modeling 前固定，禁止按 test pCR 表现筛 family。
- 关键 comparisons 使用 2,000 次 paired patient-level bootstrap；AUROC/AUPRC 为 augmented−baseline，Brier improvement 为 baseline−augmented。报告与绘图阶段只读取 aggregate metrics，不读取 patient IDs、OOF probabilities 或原始 workbook。
- Source-workbook 的独立 FTV/LD/SPH/BPE measurement 最大 cell missing count 为 0；secondary train-median+indicator scenario 是稳健性分析，不替代 primary complete-case estimand。

## 选择与解释限制

1. 这是单 cohort 的 OOF internal validation，不是 external validation；bootstrap CI 不消除 dataset shift、标签误差或治疗方案差异。
2. 多个 timing、view、target 与 metric 会产生 multiplicity。A–D 规则是预注册的保守综合，不等同于每个 cell 的正式多重检验校正，也不应把 AUROC=0.60 当临床阈值。
   此外没有预注册 equivalence margin；“未满足稳定增量”不等于效果为零或两模型约等，任何孤立 CI>0 都会阻止 B/C/D 并触发 mixed mapping。
3. Family ablation 只做 D/S/B 三个预注册单-family add-on，不穷举组合；报告的 family 排名是定位性描述，不能作为 post-test feature selection。
4. SPH 来自 FTV mask geometry、LD 是 burden measure，二者可能与 FTV 强相关；BPE 则可能来自 lesion-centered crop 之外。Residualization 缓解线性 FTV redundancy，但不能证明生物学独立或因果性。
5. T3 接近术前，预测值即便较高也可能缺乏早期决策价值。HR/HER2 probe 只说明可解码关联；profile class imbalance、calibration 与外部泛化仍需单独验证。
6. Current MRI reference 只有在同一 matched population、相同 target/timing 且 aggregate 定义可配对时才排名；不跨 n=384 primary 与 n=375 reference 作伪配对。Frozen audit 与新 classical pipeline 的 F/clinical preprocessing 和 head 不同，所以仅 N-vs-M 是同人群描述性 benchmark；clinical/FTV-augmented rows baseline-confounded，不能当作 paired incremental effect，且 primary radiomics 结论不依赖 MRI reference。

## 与 Goal 3 / Goal 5 的条件解释框架

本实验不等待其他 Goal 完成；下表只定义将来的条件解释，不把未知结果填成事实。

| 条件 | 解释 / 下一步 |
| --- | --- |
| Goal 6 classical phenotype 强，而 current MRI 弱 | current representation learning failure；优先 phenotype-aware representation learning |
| Goal 6 强，且 Goal 5 heterogeneity 强 | 优先设计显式 phenotype state，并分层检查稳定性 |
| Goal 6 弱，但 Foundation strong | foundation 捕获了 handcrafted radiomics 之外的信息 |
| Goal 6 弱，且 Foundation 也弱 | dataset 的 imaging-complementarity 可能本身有限 |

## 表格索引

- [Table 1：真实 feature inventory](../features/radiomics_feature_inventory.csv)
- [方法表：information timing contract](../information_timing_contract.csv)
- [Table 3：paired matched populations（root mirror）](../matched_population_manifest.csv)
- [Table 10：D/S/B family ablation](../metrics/family_ablation_metrics.csv)
- [补充表：feature correlation matrix](../metrics/feature_correlation_matrix.csv)
- [方法表：outer-fold hyperparameter selections](../metrics/hyperparameter_selections.csv)
- [Table 7：paired incremental effects 与 95% CI](../metrics/incremental_effects.csv)
- [Table 5：longitudinal radiomics pCR metrics](../metrics/longitudinal_radiomics.csv)
- [Table 11：LR vs RBF-SVM](../metrics/lr_vs_svm.csv)
- [Table 3：paired matched populations](../metrics/matched_population_manifest.csv)
- [Table 2：missingness 与 coverage](../metrics/missingness.csv)
- [补充表：matched current MRI pCR reference](../metrics/mri_reference_metrics.csv)
- [补充表：current MRI profile reference](../metrics/mri_reference_profile_metrics.csv)
- [补充表：n=375 matched MRI-vs-traditional pCR comparison](../metrics/mri_reference_traditional_pcr_comparison.csv)
- [补充表：n=375 matched MRI-vs-traditional profile comparison](../metrics/mri_reference_traditional_profile_comparison.csv)
- [补充表：pCR fold-level aggregate metrics](../metrics/pcr_fold_metrics.csv)
- [Table 6：C/F/N/FULL comparison（全部 OOF aggregate）](../metrics/pcr_oof_metrics.csv)
- [方法表：outer-train preprocessing audit](../metrics/preprocessing_audit.csv)
- [Table 8：HR/HER2/subtype probes](../metrics/profile_oof_metrics.csv)
- [补充表：redundancy fold-level aggregate metrics](../metrics/redundancy_fold_metrics.csv)
- [Table 9a：NONFTV→FTV redundancy](../metrics/redundancy_metrics.csv)
- [补充表：residualization fold-level aggregate metrics](../metrics/residualization_fold_metrics.csv)
- [Table 9b：FTV-residualized pCR metrics](../metrics/residualization_metrics.csv)
- [方法表：outer train/validation/test split summary](../metrics/split_summary.csv)
- [Table 4：static radiomics pCR metrics](../metrics/static_radiomics.csv)

`static_radiomics.csv` 与 `longitudinal_radiomics.csv` 是 `pcr_oof_metrics.csv` 的预定义 view 投影；Table 6 仍保留完整 `model` comparison，没有按结果挑选行。配置与规则见 [experiment.json](../configs/experiment.json)。

## 图索引

- [Figure 1：各 timing 的 pCR AUROC](../figures/timing_auroc.png)
- [Figure 2：C+F 与 C+FULL](../figures/c_f_vs_c_full_auroc.png)
- [Figure 3：配对 bootstrap ΔAUROC forest](../figures/delta_auroc_forest.png)
- [Figure 4：phenotype family ablation](../figures/phenotype_family_comparison.png)
- [Figure 5：HR/HER2 phenotype probe heatmap](../figures/hr_her2_heatmap.png)
- [Figure 6：FTV residualization](../figures/residualized_results.png)
- [Figure 7：feature correlation matrix](../figures/feature_correlation_matrix.png)

## Git / GitHub delivery provenance

| 字段 | 值 |
| --- | --- |
| branch | `PENDING` |
| commit SHA | `PENDING` |
| push status | `PENDING` |
| push error | `PENDING` |

若 `reports/delivery_provenance.json` 尚不存在或字段缺失，上述值按要求显示 `PENDING`；报告生成器不会猜测 branch、SHA 或 push 状态。
