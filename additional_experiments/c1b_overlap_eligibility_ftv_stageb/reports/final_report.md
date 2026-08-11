# C1B-H overlap eligibility amendment 与 FTV-only Stage B 最终报告

## 执行结论

本轮建立的是一个新的、outcome-free、预注册实验；它没有把上一轮 NO-GO 改写成成功。技术资格规则在任何新 cohort 统计及 Stage-B 结果之前冻结，新的 C1B-H Stage A 为 **GO（15/15 gates PASS）**，因此按预注册授权执行了完整 FTV-only 2×2 Stage B。

最终科学分类唯一选择 **Outcome D — C1B-H worse**。C1B-H 的 lesion available-support containment 达到 0.978、改善了 observability，并明显改善 Direct Grounding 的 optimization safety（N3 10/10 PASS，L3 8/10 PASS）；但在两个独立 training seeds 中，N1/N3 的 static 与 literal observed ΔFTV rank information 都在数值上明显且方向一致地弱于 matched legacy L1/L3（描述性方向，非显著性检验）。Natural-scale R²、prediction variance 和 calibration 也没有形成支持 Outcome A/B 的一致证据。

**唯一下一优先级：stronger image encoder + richer response representation。** 下一实验前先审计 excessive padding、effective spatial resolution、larger-FOV lesion dilution 与实际 budget matching；不立即修改 input，也不重新打开 preprocessing corner-case search。`FTV+LD` 与 `optimization stabilization` 均不是本轮第一优先级。

## 预注册、技术资格与 Stage A

冻结记录创建于 `2026-08-09T11:47:43Z`。通用规则为：对每名 patient 的 T0、T1、T2、T3 执行 AND，且每次 visit 均须满足 `valid_source_voxels > 0`。规则只读取 source、geometry、frozen physical grid 和 valid-source overlap；没有读取 lesion/FTV/LD/SPH/BPE、pCR、treatment subtype、loss、representation 或 performance 字段，也没有 patient-specific exception。

- 冻结计划 SHA-256：`72c440302b3992a09bc74b0a6a59b8bf11d0e20835322f58cbbd5bd43608d588`；见 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 和 [preregistration lock](../configs/preregistration_lock.json)。
- Stage-A finalizer 观察到 `stage_b_artifacts_before_gate=0`；阈值未放宽。
- Candidate population：948 patients、3792 visits；其中 3791 visits 为 positive-overlap，1 visit 为 zero-overlap。
- Eligible longitudinal population：947 patients、3788 visits；排除 1 patient，aggregate reason 为 `ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT`。
- 3788 而不是 3791 的原因：四访 AND 排除该 patient 后，其另外 3 个 positive-overlap visits 也随 longitudinal patient 一并退出。
- 375 patients / 1500 visits 是 formal FTV analysis subset，不是技术资格分母；Stage B matched population 为 947（808 fold-assigned + 139 upstream-authorized train-only）。

Stage A 对 eligible cohort 完成 947/947 patients、3788/3788 visits 的 DCE7 cache；completion、finite、shape、phase、orientation、grid 与 nonconstant fractions 全部为 1.0。完整 candidate population 的 true array-reordered RAS+ fraction 为 1.0；交集后 formal containment 为 0.978，physical FTV retention Q05 为 1.0，1500 formal visits 中 1486 为 grounding-observable，14 个 loss-side mask 为 false。geometry、mask 与 support sidecars 均未进入 model tensor。

Stage A 为 GO，哨兵 SHA-256 为 `0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb`；见 [STAGE_A_GO.json](../STAGE_A_GO.json)、[Stage-A gate report](stage_a_gate_report.md)、[technical eligibility amendment](technical_eligibility_amendment.md) 和 [Table 1](../metrics/table1_technical_eligibility_stage_a_qc.csv)。

## Stage B 完整性与分析口径

Stage B 精确执行 2 seeds × 5 folds × 4 arms = 40 cells。所有 arms 均为 physical batch 4 × accumulation 8 = logical B32；272/272 training-history rows finite/non-collapsed，40/40 selected checkpoints finite/non-collapsed，selection 和 checkpoint 的 80 个 test-use flags 全为 false。Training seed 是主要独立单位；fold、patient visit 和 endpoint 不是独立重复。两个 seed 仅构成 pilot，不支持最终 multiseed robustness 声称。

完成链为：

- matrix `COMPLETE`，SHA-256 `0adbcf7daf74f31e70c64c1ec9a5bb259411792fb0dfa4d093ee9d3e3210b4a2`；
- 40/40 frozen response-state features，feature completion SHA-256 `f8bc1a158c93c0563b11e46cb02c4b0ef5681048febd94ab6d674d3ea4fdc40d`；
- 40/40 fixed Ridge probes，postprocessing completion SHA-256 `4a599f5d76482677056f9df11e46faa1b8d4f277eedabb63d60306531e841558`；
- aggregation summary `COMPLETE`，SHA-256 `2c70c429d7b32640160f8ffbbf9b3f3f7b991838227a964387bd2f4090e445c4`；见 [aggregation summary](../metrics/stage_b_aggregation_summary.json)。

Static 主结果使用 `analysis_scope=primary_measurement_valid`、natural scale、每 seed 五折 pooled OOF；macro 是 T0–T3 endpoint metrics 的非加权均值。Observed ΔFTV 严格为 natural `FTV_end − FTV_start`，对 T0→T1、T1→T2、T2→T3 分别分析；训练与 checkpoint selection 未使用 Δ supervision。Ridge 的 scaler 与 alpha selection 仅使用 outer train/validation，outer test 每 cell 仅预测一次。

## 四臂主要结果

下表报告每个 training seed 的五折 pooled OOF macro；`ρ` 为 Spearman，R² 为 natural scale。

| Seed | Arm | Static ρ | Static R² | Literal ΔFTV ρ | Literal ΔFTV R² |
|---:|:---:|---:|---:|---:|---:|
| 2026 | L1 | 0.531285 | −0.066594 | 0.207506 | 0.014871 |
| 2026 | L3 | 0.590506 | 0.061715 | 0.279489 | 0.029853 |
| 2026 | N1 | 0.230515 | −0.008670 | 0.074489 | −0.061008 |
| 2026 | N3 | 0.267180 | −0.015042 | 0.120933 | −0.010422 |
| 3026 | L1 | 0.478733 | 0.035408 | 0.189204 | 0.015713 |
| 3026 | L3 | 0.558577 | 0.100438 | 0.320390 | 0.028779 |
| 3026 | N1 | 0.220304 | −0.009900 | 0.069542 | −0.001839 |
| 3026 | N3 | 0.255120 | 0.031168 | 0.096595 | 0.011320 |

完整 endpoint、RMSE、MAE、baseline gain、variance 与 descriptive calibration 见 [Table 2](../metrics/table2_static_ftv.csv) 和 [Table 3](../metrics/table3_literal_observed_delta_ftv.csv)。

### 配对效应与 Difference-in-Differences

| Seed | Task / metric | N1−L1 | L3−L1 | N3−N1 | DiD `(N3−N1)−(L3−L1)` |
|---:|---|---:|---:|---:|---:|
| 2026 | static ρ | −0.300770 | +0.059221 | +0.036665 | −0.022556 |
| 2026 | static natural R² | +0.057924 | +0.128309 | −0.006372 | −0.134681 |
| 2026 | ΔFTV ρ | −0.133017 | +0.071983 | +0.046444 | −0.025539 |
| 2026 | ΔFTV natural R² | −0.075879 | +0.014982 | +0.050586 | +0.035604 |
| 3026 | static ρ | −0.258429 | +0.079844 | +0.034816 | −0.045028 |
| 3026 | static natural R² | −0.045308 | +0.065030 | +0.041069 | −0.023962 |
| 3026 | ΔFTV ρ | −0.119662 | +0.131187 | +0.027053 | −0.104134 |
| 3026 | ΔFTV natural R² | −0.017552 | +0.013065 | +0.013159 | +0.000094 |

两 seed 的 N1−L1 static Spearman 均大幅为负；10/10 macro folds 和 40/40 endpoint-fold comparisons 都为负。ΔFTV Spearman 的 N1−L1 在 seed 2026 为 5/5 folds 负、seed 3026 为 4/5 folds 负。N3−N1 的 rank gain 为正但小于 L3−L1；static Spearman DiD 正向仅 2/5 与 1/5 folds，ΔFTV Spearman 同样仅 2/5 与 1/5。Static R² DiD 在两个 seed 均仅 1/5 folds 为正；ΔFTV R² 为 3/5 与 1/5。完整 sensitivity 见 [fold-level Table 5](../metrics/table5_fold_level_sensitivity.csv)，seed-level 主比较见 [Table 5](../metrics/table5_difference_in_differences.csv)。

### Natural-scale variance 与 calibration

Static prediction/target variance ratio 的 macro 端点均值：L1/L3 为 0.112–0.234，N1/N3 仅 0.025–0.038；descriptive calibration slope 分别为 0.104–0.124 与 0.031–0.058。C1B arms 的预测明显更压缩，不能用 transformed-space correlation 掩盖。ΔFTV 的所有 arms 也弱：variance ratio 0.048–0.102、slope 0.021–0.059。Selected representation std 范围为 0.160669–0.689166，全部高于 0.05 collapse 门槛；四臂均值为 L1 0.502、L3 0.519、N1 0.570、N3 0.541。因此这是 FTV-readout information / calibration 弱，而不是数值 collapse。

## Optimization safety

预注册安全条件为 selected validation state loss `<= 1.05 ×` paired no-grounding baseline；恰好 5% 计 PASS。

- N3：10/10 PASS，mean degradation −2.3084%，最大 +4.3145%。
- L3：8/10 PASS，mean degradation +3.4325%，最大 +31.0157%。
- 两个 L3 failures 均来自 seed 3026：fold 0 为 +7.484999%，fold 3 为 +31.015701%；两者 finite/non-collapsed，按冻结规则保留为 `fallback_base_gate_failed`，没有从 representation analysis 中删除。
- 按 seed，2026 的 L3/N3 mean degradation 为 −0.8952%/−5.0202%，3026 为 +7.7602%/+0.4033%；optimization DiD 分别为 −4.1249 和 −7.3570 percentage points，负值代表 C1B 更安全。

因此 operational matrix completion 为 PASS，但全矩阵 optimization-safety 科学结果不是“全通过”：L3 有 2 个预注册失败。精确 40-cell 结果见 [Table 4](../metrics/table4_optimization_safety.csv)。

## 对 14 个问题的明确回答

1. **新 technical eligibility rule 是否在任何 Stage-B 结果之前冻结？** 是。计划于 `2026-08-09T11:47:43Z` 冻结，SHA-256 为 `72c440…d588`；Stage-A finalizer 记录 `stage_b_artifacts_before_gate=0`。规则也在新 cohort statistics 之前预注册。
2. **实际 eligible population 是多少？** 947/948 patients、3788 longitudinal visits。375-person formal FTV subset 和 808 fold-assigned population 都不是 eligibility 分母。
3. **Zero-overlap technical exclusions 有多少？** 1 patient、1 zero-overlap required visit；通用 aggregate reason 为 `ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT`。四访 AND 同时移除该 patient 的另外 3 个 positive visits。
4. **Stage A 是否 100% model-ready？** 是：947/947 patients、3788/3788 visits cache 完成，全部 cache QC fractions 为 1.0，15/15 gates PASS，containment 0.978，FTV retention Q05 1.0。
5. **N1 相对 L1 有什么变化？** 明显变差。Static macro ρ 下降 −0.300770/−0.258429；ΔFTV macro ρ 下降 −0.133017/−0.119662。Static natural R² 的 N1−L1 为 +0.057924/−0.045308，方向不一致；ΔFTV natural R² 为 −0.075879/−0.017552，两 seed 均下降。Static variance ratio 与 calibration slope 在两 seed 均更压缩；这些结果不能抵消一致的 rank-information loss。
6. **N3 相对 N1 是否稳定改善 static FTV？** Spearman 在两个 seed 均小幅上升 +0.036665/+0.034816，但 fold 方向仅各 3/5 为正；natural R² 为 −0.006372/+0.041069，方向不一致。因此不是稳定、足以支持 observability bottleneck 的改善。
7. **Natural-scale FTV R² 是否明显改善？** 否。N3−N1 static R² 一负一正且均弱于 L3−L1；N3 absolute static R² 也低于同 seed L3。预测方差和 calibration 显示 C1B arms 更严重压缩。
8. **Observed ΔFTV 是否改善？** Grounding 对 N arms 有小幅改善：ρ +0.046444/+0.027053，R² +0.050586/+0.013159；但 N1/N3 的 absolute macro ρ 均低于 matched L1/L3 counterpart，natural R² 也均未超过 matched counterpart。只有 seed 2026 的 ΔFTV R² 显示 N3−N1 明显大于 L3−L1，seed 3026 几乎相等，不能称跨 seed 稳定。
9. **N3−N1 是否强于 L3−L1？** 总体否。Static ρ、static R² 和 ΔFTV ρ 均不支持；ΔFTV R² 仅一个 seed 支持、另一个相等。Safety 则明确强于：N3 10/10 对 L3 8/10。
10. **Difference-in-Differences 支持什么？** Representation DiD 大多为负，说明 C1B 没有放大 grounding 的 static/dynamic readout gain；只有 ΔFTV R² 出现一强一近零的正 DiD。Optimization degradation DiD 在两个 seed 均为负，支持 C1B 提高 grounding compatibility，但不支持其提高总体 quantitative faithfulness。
11. **C1B 是否改善 base optimization safety？** 是。N3 10/10 PASS，L3 8/10；两个 seed 的 mean-degradation DiD 均为负。该结论与 representation 负结果并存。
12. **Legacy partial crop 是否可以解释此前 G3 的部分问题？** 只得到有限、机制层面的支持：本轮 C1B 更安全，说明 partial evidence/global target mismatch 可能是旧 grounding instability 的一个贡献因素；但 C1B representation 明显更差，不能解释或修复旧 G3 的全部 calibration/readout 问题，更不能追溯宣称唯一因果。旧 G3 仅作外部历史：5-seed static/ΔFTV ΔSpearman 均值约 +0.0572/+0.0894、ΔFTV ΔR² 约 +0.06045、safety 17/25；原始单-seed natural static ΔR² −0.0676，且有一 fold +9.5934% degradation。旧结果不与本轮合并推断。
13. **当前主要 bottleneck 是什么？** 在当前 matched encoder/budget 下，是 image encoder 与 response representation 对 larger-FOV C1B 信号的提取能力，而不是数值 collapse 或主要的 grounding optimization instability。结果与 padding、effective-resolution loss、lesion dilution 或 budget mismatch 相容，但本轮不能区分这些机制。
14. **下一步三选一是什么？** 唯一第一优先级是 **stronger image encoder + richer response representation**。`FTV+LD` 不优先，因为 observability repair 未产生更忠实的 FTV representation；`optimization stabilization` 不优先，因为 N3 已 10/10 安全，且它不能解释弱 readout。

## Outcome A–D 决策

冻结计划没有给“明显”或“≈”的事后数值阈值。本表依据两个 seed 的精确 effect、方向与 fold sensitivity 作描述性唯一映射。

| Outcome | 状态 | 证据判断 | 对应优先级 |
|---|---|---|---|
| A — observability bottleneck supported | NOT SELECTED | Safety 条件成立，但 static R²/ρ 与 ΔFTV ρ 的 representation 条件不成立；仅一个 seed 的 ΔFTV R² 有较强正 DiD。 | FTV+LD 不优先 |
| B — observability improved, optimization bottleneck remains | NOT SELECTED | Representation 未明显改善，且 N3 safety 不是“与 L3 类似或仍差”，而是 10/10 PASS。 | optimization stabilization 不优先 |
| C — input fix not representation-limiting | NOT SELECTED | N1 并不约等于 L1；static/dynamic rank information 在两个 seed 都明显下降。 | — |
| **D — C1B-H worse** | **SELECTED** | Absolute static/ΔFTV Spearman 在两 seed 均数值上明显且方向一致地低于 matched legacy；N3 natural R² 在两任务、两 seed 均低于 L3，且 static prediction 更压缩。 | **stronger encoder + richer response representation** |

## 与旧结论的边界、执行异常与限制

上一轮 [STAGE_A_NO_GO.json](../../c1b_model_ready_ftv_sanity/STAGE_A_NO_GO.json) 仍为 immutable，SHA-256 `ad2604d35c9fca645f6487c7decf297a0c8f0711136973491d537ac42aa8f080`；独立 provenance audit 的 [AUDIT_NOT_REPAIRABLE.json](../../zero_overlap_provenance_audit/AUDIT_NOT_REPAIRABLE.json) 仍为 immutable，SHA-256 `042dd629fdb10a7b08bfdeceaa6cf51d9d9e7e713fa5a86027a4d569640d0ffd`。本轮没有删除坏病例后宣布旧 run 成功，而是在 independent no-authoritative-repair 结论之后，执行新规则与新实验。

Stage-A cache 复用曾短暂对 262 个旧 cache 文件创建 hardlink；在 Stage-A finalizer 前已逐个 copy/reflink、完整 byte compare 并 atomic replace。最终 shared inode 为 0、link count 均为 1，262/262 bytes、SHA-256、size 与 mtime 不变；但 inode ctime 可能改变，不能声称旧 cache 的所有 filesystem metadata 从未变化。证据见 [cache independence verification](../metrics/cache_independence_verification.json)。

首次 formal root 在第一批 L1 optimizer step 前因审计器 expected-count bug fail-fast；该 root 仅含 preflight/空目录，没有 selection、history 或 checkpoint，且未进入 features、probes 或 aggregation。随后从新的空 `formal_4x8_restart1` root 完整重启。Exact B32 SIGReg、fail-fast 修正及 completion-chain hardening 见 [bug-fix ledger](bug_fix_ledger.md)。这些修正发生在相应正式结果之前，没有改变四臂、样本规则、loss 权重、阈值或 frozen scientific endpoints。

本轮没有进行 LD training。可选 secondary pCR frozen readout 未运行，且 pCR 不得改变本轮 Outcome。没有用 patient bootstrap 或 fold 数量制造显著性；所有“改善/变差”均是两 training-seed pilot 的精确描述性方向。

本报告落盘后的最终 [public-artifact privacy gate](../metrics/public_artifact_privacy_gate.json) 为 PASS：扫描 28 个公开文本工件，与 948 个 private identifier values 交叉检查，identifier/path findings 为 0，private-permission findings 为 0。

## 主要表与图

- Tables：[Table 1](../metrics/table1_technical_eligibility_stage_a_qc.csv)；[Table 2](../metrics/table2_static_ftv.csv)；[Table 3](../metrics/table3_literal_observed_delta_ftv.csv)；[Table 4](../metrics/table4_optimization_safety.csv)；[Table 5](../metrics/table5_difference_in_differences.csv)；[fold sensitivity](../metrics/table5_fold_level_sensitivity.csv)。
- Stage A Figures：[Figure 1](../figures/01_technical_eligibility_flow.png)；[Figure 2](../figures/02_valid_source_overlap_distribution.png)；[Figure 3](../figures/03_cache_completion_qc.png)。
- Stage B Figures：[Figure 4](../figures/04_static_ftv_spearman.png)；[Figure 5](../figures/05_static_ftv_natural_r2.png)；[Figure 6](../figures/06_static_ftv_predicted_vs_true_natural.png)；[Figure 7](../figures/07_literal_delta_ftv_spearman.png)；[Figure 8](../figures/08_literal_delta_ftv_natural_r2.png)；[Figure 9](../figures/09_state_loss_degradation_heatmap.png)；[Figure 10](../figures/10_representation_std.png)；[Figure 11](../figures/11_grounding_difference_in_differences.png)；[Figure 12](../figures/12_representative_training_curves.png)。Figure 9 色标在 +10% 饱和，因此 +31.0157% 单元的精确值以 Table 4 为准；Figure 12 只是 representative curves，不替代 40-cell evidence。

科学链条保持为：legacy fixed voxel crop → poor lesion observability → C1B physical crop → 0.978 available-support containment → full DICOM/orientation validation → one catastrophic longitudinal coordinate case → independent provenance audit → no authoritative repair → pre-registered technical eligibility amendment → model-ready C1B cohort → FTV-only causal representation sanity。答案是：**more observable image state 提高了 optimization compatibility，但在当前 encoder/budget 下没有使 Direct Grounding 的 response representation 更定量忠实。**
