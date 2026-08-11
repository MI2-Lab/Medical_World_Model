# Spatial Heterogeneity / Phenotype Pooling Audit 最终报告

## 结论先行

本实验的 Stage-A 科学分类为 **B. `PHENOTYPE SPATIALLY LOCALIZED`**。四个预注册 Gate：A=FAIL，B=FAIL，C=PASS，D=FAIL。该分类只由冻结 Stage A 决定，Conditional Stage B 不回写诊断结论。

## 预注册修订披露

原始预注册在任何临床标签表解析、Stage-A probe 拟合或 Stage-B 启动之前的 geometry QC 中发现：四次 source-authorized 患者仍为 375，但经过固定 LOCAL 支持约束后，四次 CORE 均有效的 Figure-8 去标识化代表候选为 373，而不是原先硬编码的 375。修订仅校正 Figure 8 代表选择；FTV/pCR 的 375 人群、probe、Gate 与科学分类合同均未改变。修订前生成的 13 个 cache/oracle/feature/log 工件均按公开哈希台账丢弃且禁止复用；当时 completed feature cells=3，clinical label table parsed=false，Stage-A probe fit=false，Stage-B started=false。完整修订记录由 `PREREGISTRATION_AMENDMENT.json` 固定。

## 实现勘误披露

在上述 amended preregistration 下，cache、Oracle、20 个 feature cells 与去标识化代表资产已完成；独立只读复核验证了 20 个 cells，最大 P1 projection parity absolute difference=0.0。Feature-matrix completion validation 随后仅因调用 spatial feature loader 时遗漏 keyword-only `seed`/`arm`/`fold` 参数而停止，且 completion marker created=false。此修复只把三个身份参数作为 keyword arguments 传入；scientific、representative 与 causal-Oracle contracts 均未改变。失败时的 65 个工件（307,933,315 bytes）由 `PREREGISTRATION_IMPLEMENTATION_ERRATUM.json` 的公开哈希台账绑定，均在实现勘误 refreeze 前丢弃并禁止复用。当时 clinical label table parsed=false，Stage-A probe fit=false，Stage-A result created=false，Stage-B started=false。

## 实现兼容性勘误 2 披露

第一次实现 refreeze 后，cache/Oracle、20 个 feature cells、代表资产与 feature-matrix completion marker 均完成。Stage A 解析了冻结 fold/pCR labels、clinical phenotype labels 与 FTV；唯一落盘的 Stage-A 文件是无标签派生值的 Table 1。首个 `seed_2026/LOCAL0/fold_0`、`T0`、`P1` cell 中，HR/HER2 两个 binary tasks 的 14 次候选拟合、324 行预测与 2 行超参数只存在内存中。首个 subtype `C=0.0001` 拟合因 sklearn 1.8 禁止 legacy bare multiclass liblinear 而在拟合前失败；没有 patient-level prediction、label-derived public metric、Table 2+、hyperparameter table、gate、authorization、run summary 或 Stage B 工件落盘。

兼容修复通过 sklearn `_fit_liblinear` 得到 binary OvR rows，并严格复现 legacy sigmoid 与逐行概率归一化；它不是 generic child-balanced OvR。solver、penalty、class-weight、C grid、选择/tie、outer folds/populations 与 scientific/representative/causal-Oracle contracts 均不改变。local runner 仅抑制 sklearn 1.8 重复的 `penalty='l2'` deprecation `FutureWarning`，不改变 estimator；`ConvergenceWarning` 继续 fail-closed。`PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json` 精确绑定失败时 67 个工件（307,938,585 bytes；record-set SHA-256 `97768d153498c4fb953184ed6677e61ff4dc083f2f98362fafa360c896f39484`）；全部在 schema-5 refreeze 前丢弃且禁止复用。

## 实现验证勘误 3 披露

兼容性 refreeze 后，Stage A 已完整结束：691,412 行 private OOF prediction、1,256 行注册 metric（不含 pooling contract）与 6,280 行 hyperparameter selection 均通过 grid、OOF/fold isolation、hash、privacy、mode、authorization 与 run-summary 复核。默认 CSV parser 在 Gate 重算对象中造成 26 个纯数值舍入差异，绝对差范围为 [5.5511151231257827e-17, 1.1102230246251565e-16]；`float_precision='round_trip'` 精确恢复已发布 Gate 对象及 SHA-256 `1b1063f6c0c545b57738050738d7581157e1af31a9ebfaa0ec87d520c9b16ded`。A/B/C/D 决策与 `PHENOTYPE SPATIALLY LOCALIZED` 分类均未改变。

同一次预 refreeze 审计修正四个尚未生成公开 figure/report 工件的呈现合同：Q2 另列严格 16 对 matched-375 pCR P2−P1；Figure 7 将三种 variant 在 `full_808` 与 `ftv_complete_375` 分成六条曲线；Q12 只用配置的 matched primary pCR population 作主汇总而不跨人群平均；Q8/Q9 从 Gate-C supporting records 导出 endpoint-specific 支持并明确 phenotype 与 pCR 的边界。这些修正不改变模型、表格、metric、Gate、authorization 或科学合同。Stage-B folds 0--2 已进入 epoch 1 执行，但在任何 epoch 完成前中断，completed epochs=0，Stage-B files/checkpoints/results 均为 0；六个目录仅是 artifact-empty side effects。`PREREGISTRATION_IMPLEMENTATION_ERRATUM_3.json` 精确绑定当时 83 个文件（409,345,148 bytes；record-set SHA-256 `01b6f348c78f800b044bce5659fb8b029956299fa4e9585e75d6a940cadb1ace`），全部在 schema-6 refreeze 前丢弃且禁止复用。

## 十二个问题的明确回答

1. **Mean pooling 是否丢失 heterogeneity information？** 未获得预注册阈值证据（支持来源：none）。Gate A 检验 P3−P1 的 phenotype/matched-pCR 信号；Gate B 独立检验 P3 在 C+F 之外且优于 C+F+P1 的互补性。P3−P1 phenotype AUROC：平均 -0.002，范围 [-0.043, +0.054]；最大值位于 seed=3026, arm=LOCAL0, view=T3, target=HR, population=full_808。
2. **Std 是否有独立价值？** 未为 P2 单独预注册 Gate，因此这些描述性配对结果不能建立稳定的独立 SD 价值。P2−P1 phenotype AUROC：平均 +0.003，范围 [-0.052, +0.048]；最大值位于 seed=3026, arm=LOCAL0, view=T3, target=HR, population=full_808。独立的 matched-375 pCR P2−P1（严格 16 对）：平均 -0.010675，范围 [-0.057667, +0.085763]；最大值位于 seed=3026, arm=LOCAL3, view=T0-T3, target=pCR, population=ftv_complete_375。它们回答可解码性，不把单次正增益解释为因果或泛化证明。
3. **Mean+Std 是否优于 Mean？** 未满足直接 Gate-A 或互补 Gate-B 的双-seed标准（none）；Gate B 通过时即表示 C+F+P3 同时优于 C+F 与 C+F+P1，不能因 Gate A 失败而忽略。
4. **HR/HER2/subtype 是否改善？** 以两个 seed 均满足 P3−P1 AUROC ≥ +0.03 分别判定：HR=NOT_SUPPORTED（0 个 arm/view 比较）；HER2=NOT_SUPPORTED（0 个 arm/view 比较）；subtype_4class=NOT_SUPPORTED（0 个 arm/view 比较）。HR 不进入 Gate A；HER2/subtype disjunct=FAIL。HR 全部成对结果：平均 -0.002，范围 [-0.043, +0.054]；最大值位于 seed=3026, arm=LOCAL0, view=T3, target=HR, population=full_808。
5. **pCR 是否改善？** matched 375 的独立双-seed pCR disjunct=FAIL（pCR=NOT_SUPPORTED（0 个 arm/view 比较））；全部 P3−P1 成对结果：平均 -0.003，范围 [-0.055, +0.061]；最大值位于 seed=2026, arm=LOCAL0, view=T0-T3, target=pCR, population=ftv_complete_375。该结论不由 phenotype disjunct 代替。
6. **是否有 FTV 之外的信息？** Gate B=FAIL。C+F+P3−C+F：平均 -0.036，范围 [-0.064, -0.004]；最大值位于 seed=3026, arm=LOCAL3, view=T0-T1, target=pCR, population=ftv_complete_375。
7. **Core 还是 peritumoral 含更多 phenotype signal？** 仅在 HR/HER2/subtype 内，各区域与其同一 oracle_pair complete-case 人群的 FIXED_P3 比较；最大 matched uplift 及平均配对增量排序为：PERI10: ΔAUROC=+0.004, CORE_PERI: ΔAUROC=+0.002, CORE: ΔAUROC=+0.001, PERI20: ΔAUROC=-0.000, LOCAL_REST: ΔAUROC=-0.011。区域间使用不同 variant-specific cohorts，因此 absolute core-vs-peri AUROC 排名不可识别，也不作该排序。
8. **Oracle 是否强于 mask-free fixed local？** Gate C=PASS，支持比较数=1：arm=LOCAL0, comparison=PERI20 vs reference=FIXED_P3, target=pCR, view=T0-T1, population=oracle_pair_PERI20 (seed2026=+0.03567753001715257; seed3026=+0.033276157804459694)。其中 HR/HER2/subtype 支持数=0，pCR 支持数=1。每个比较都在相同、variant-specific complete-case 人群并限制于同一 64-mm LOCAL support。
9. **当前瓶颈是什么？** Stage-A 唯一分类为 B. `PHENOTYPE SPATIALLY LOCALIZED`；Gate D=FAIL。当前 B 标签狭义地由上述 pCR supporting comparison 驱动；HR/HER2/subtype 没有 Gate-C supporting comparison。该标签的 Gate-C 支持明细为 arm=LOCAL0, comparison=PERI20 vs reference=FIXED_P3, target=pCR, view=T0-T1, population=oracle_pair_PERI20 (seed2026=+0.03567753001715257; seed3026=+0.033276157804459694)。
10. **是否需要 Response–Phenotype factorized state？** 当前证据不足以把它列为首选；应先处理定位或表征问题。
11. **是否需要 foundation encoder？** 本审计不要求立即更换；应先按当前分类完成更大样本/多 seed 确认。
12. **Stage B 是否授权、结果如何？** 已授权；表 8 状态为 COMPLETE。相对同 seed/view/target/population 的 Stage-A LOCAL3/P1 基线（Stage-B 为 dual-state arm）：phenotype ΔAUROC 平均 +0.011，范围 [-0.020, +0.042]；pCR primary population `ftv_complete_375` ΔAUROC 平均 -0.051，范围 [-0.095, -0.025]；pCR Brier 改善平均 -0.042。

## 方法与解释边界

- Stage A 使用已选择且 test-blind 的 LOCAL0/LOCAL3 checkpoint，共 2 seeds × 5 folds；encoder 不重训。
- 实际 pre-pooling tensor 为 128×14×22×20；P1/P2/P3/P4/P5 均在同一固定 64-mm fractional LOCAL support 上计算。
- 所有 scaler、临床编码、FTV residualization、C 与阈值选择只在 outer train/validation 内完成；outer test 只预测一次。
- Oracle 只做机制定位，所有区域在 feature grid 上再限制到同一 500-cell LOCAL support；大 receptive field 与粗 Z 方向分辨率仍限制精细边界解释。
- 本报告中的“改善”是冻结 OOF 诊断结果，不等于独立外部验证或统计显著性。

## 表格

- [表 1：table1_pooling_contract.csv](../metrics/table1_pooling_contract.csv)
- [表 2：table2_phenotype_probes.csv](../metrics/table2_phenotype_probes.csv)
- [表 3：table3_mri_only_pcr.csv](../metrics/table3_mri_only_pcr.csv)
- [表 4：table4_clinical_ftv_incremental.csv](../metrics/table4_clinical_ftv_incremental.csv)
- [表 5：table5_residualized_mri.csv](../metrics/table5_residualized_mri.csv)
- [表 6：table6_longitudinal_heterogeneity.csv](../metrics/table6_longitudinal_heterogeneity.csv)
- [表 7：table7_oracle_regions.csv](../metrics/table7_oracle_regions.csv)
- [表 8：table8_stage_b.csv](../metrics/table8_stage_b.csv)

## 图

- [图 1：01_pooling_schematic.png](../figures/01_pooling_schematic.png)
- [图 2：02_phenotype_auroc_by_pooling.png](../figures/02_phenotype_auroc_by_pooling.png)
- [图 3：03_pcr_auroc.png](../figures/03_pcr_auroc.png)
- [图 4：04_beyond_ftv_delta.png](../figures/04_beyond_ftv_delta.png)
- [图 5：05_mean_vs_std.png](../figures/05_mean_vs_std.png)
- [图 6：06_core_peritumoral_comparison.png](../figures/06_core_peritumoral_comparison.png)
- [图 7：07_longitudinal_heterogeneity.png](../figures/07_longitudinal_heterogeneity.png)
- [图 8：08_representative_spatial_activation_statistics.png](../figures/08_representative_spatial_activation_statistics.png)

## 可复现性与交付

- Branch：`feature/spatial-heterogeneity-phenotype-audit`
- Preregistration revision：`2`
- Original preregistration commit SHA：`116b34f0eeec7485ead4d076517dcdfdf10960e8`
- Original preregistration lock：`7ef5eb028d6dbc1e2c01d01cc86787d89a7156ffb9c6e7b5b61b9fe22591ca21`
- Preregistration amendment：`a8af41acc6d89bab956430fe35123a8520898d1443dfe1431b75905e13559690`
- Prior amended preregistration commit SHA：`cdc7a57bf1ff373d97a97f51817ea83abe75d7e3`
- Prior amended preregistration lock：`12e3c046f108d601c99fd354745fc5620e3ab234a72f307a2b1529063b7be0c4`
- Preregistration implementation erratum：`ea49551e49a57bb7ddc52bc5ab841fab3cc9c3e1f1a9959d4f8a621d0dae9662`
- Prior implementation refreeze commit SHA：`01ecd8a4101a3a122bc58d148960f1e36f57720d`
- Prior implementation refreeze lock：`6063df14db751d9ad2f25af57e48b99a3d2282571f75c4683a24deb5a2762ce5`
- Preregistration implementation erratum 2：`db9eeda0895f65caadd508c1ed0ab499862463f0e0fe287022001ab0a25a5e35`
- Prior compatibility refreeze commit SHA：`226003f31f876c314e7c1e31092a4bf816aa89e7`
- Prior compatibility refreeze lock：`7d1b0dc1a789a83510e0e5e48b926ed744c152f009527aa0393f43601925575b`
- Preregistration implementation erratum 3：`e4d9ebcf85f611e9fcd3cd9e8973f2b25acc4e8b837d5c9a47bfcb59c01b934b`
- Preregistration lock：`bf838056dcdf2286c6365468fec274137f482adb188ad82620cceadcae393a1a`
- Preregistration commit SHA：`7de344f0b7ac3393d07616e538fdcafd487944c3`
- Active implementation-refrozen preregistration commit SHA：`7de344f0b7ac3393d07616e538fdcafd487944c3`
- Preregistration base HEAD：`226003f31f876c314e7c1e31092a4bf816aa89e7`
- Experiment commit SHA：`PENDING`
- GitHub push status：`PENDING`
- 原始 MRI、patient-level feature/prediction 与 checkpoint 均未纳入版本控制；公开表不含 patient identifier。
