# C1B Spatial Information / Pooling Bottleneck Audit 最终报告

## 执行结论

本轮在不重新训练任何模型的前提下，对 2 seeds × 4 arms × 5 folds 的 40 个 frozen checkpoints 执行了预注册 spatial pooling audit。结论是：C1B-H final spatial feature map 中保留了 substantial lesion-related rank information，但现有 Global Average Pooling（GAP）丢失了其中很大一部分；固定、mask-free 的 64-mm local readout 已能恢复大部分 legacy deficit。

FINAL_CLASSIFICATION: A POOLING BOTTLENECK

NEXT: Local–Global Response State Pilot

这是预注册门下的 **SUPPORTED IN PILOT**，不是多 seed 稳健性或统计显著性声明。两个 seed 的 N1 PORACLE−P0 static macro Spearman 分别为 +0.274919 / +0.283783，恢复 91.4% / 109.8% 的 matched legacy deficit；PLOCAL+GLOBAL 分别提高 +0.216924 / +0.207921，恢复 72.1% / 80.5%。PVALID 只提高 +0.057579 / +0.008034，而且没有降低 padding/valid-source decodability，因此 padding/geometry 不是本轮主要解释。

按冻结规则，final PORACLE 已在两个 seed 同时通过 Strong Oracle Recovery gate，所以 secondary S3 未触发，且任何 S3 feature/probe 执行均被机器 gate 禁止。Oracle 仅是不可部署的 lesion-support diagnostic upper bound；下一步不是使用 segmentation mask，而是测试固定 local state 与 global context state 的无 mask 组合。

## 预注册与执行完整性

- [实验计划](../EXPERIMENT_PLAN.md) SHA-256：`3120b7df0d47199c02227821913e536f58945691a9e5358fcd289d6abe61a477`。
- [冻结配置](../configs/audit.json) SHA-256：`ad19c58dbbd4ed4ba694686608c19f9968a2339b29f395fea620ffb41038d227`。
- [预注册锁](../PREREGISTRATION_LOCK.json) SHA-256：`0f51dd38539d2e2e9e0b8c5e2975b17f24328c9b611132deb7cbcf683d2fd57c`。
- 40/40 checkpoints 严格加载；没有 encoder、JEPA、transition、grounding 或其他模型重训练。
- 正式 feature matrix 为 40 checkpoint cells、180 deterministic pooled assets；正式 probe matrix 为 180 cells，其中 160 个 primary、20 个 N-arm `PLOCAL+PVALID` secondary sensitivity。
- [P0 state equivalence gate](../metrics/p0_equivalence_gate.json) 为 PASS：40 × 808 × 4 × 192 = 24,821,760 个元素全部 bitwise equal，最大绝对误差和平均绝对误差均为 0；冻结容差为 `rtol=1e-5, atol=1e-6`。
- [P0 probe replication gate](../metrics/p0_probe_replication_gate.json) 为 PASS：560 个 selection cells、41,704 行 OOF predictions、144 行 pooled natural metrics 与 immutable Stage-B 输出精确一致，最大 prediction/metric 差异均为 0。
- [S3 trigger authorization](../metrics/s3_trigger_authorization.json) 为 `NOT_TRIGGERED_FINAL_ORACLE_STRONG`，`s3_execution_authorized=false`；S3 feature/probe asset 均为 0。
- [Prospective gates](../metrics/prospective_gates.json) 为 COMPLETE，且聚合期间没有 probe refit。

Natural 指标均先合并五个 outer-test folds，再计算 pooled OOF metric。Transformed 指标只作 outer-fold summaries；不同 fold 的 train-fitted transform 没有被错误拼接。Static macro 是 T0–T3 四个 pooled endpoint metrics 的等权均值；observed ΔFTV macro 是三个 transition endpoint metrics 的等权均值。

## Feature-map 合同

Primary final feature 是 GAP 的直接输入，即完整 `encoder.features[3]` residual-block 输出：

| Input | Input shape ZYX | Final map | Channels | Input jump | Median physical step XYZ | Theoretical RF |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 32×96×96 | 4×12×12 | 128 | 8 voxels | 5.469×5.469×16.0 mm | 47 voxels |
| C1B-H | 112×176×160 | 14×22×20 | 128 | 8 voxels | 7.2×7.2×16.0 mm | 47 voxels |

C1B final theoretical receptive-field footprint 为约 42.3×42.3×94.0 mm；47³ 是卷积/残差路径的 theoretical support union，不是 learned effective-RF 权重。PVALID/PORACLE 使用 exact RF occupancy（kernel 47、stride 8、padding 23、固定分母）；PLOCAL 使用 feature sampling-cell 与 frozen central 64×64×64-mm window 的 fractional overlap，不按 lesion 或 outcome 移动窗口。完整合同见 [Table 1](../metrics/table1_feature_map_contract.csv)。

S3 的预定义合同为 `encoder.features[2]` raw pooled 64-D、jump 4、RF 23；C1B/legacy map 分别为 28×44×40 / 8×24×24。它没有可用的 frozen 64→192 projection，且本轮未触发、未执行。

## Primary N1 结果

下表均为五折 pooled OOF、natural scale、primary measurement-valid population；`ρ` 为 Spearman。

| Seed | Pooling | Static ρ | Static R² | ΔFTV ρ | ΔFTV R² |
|---:|---|---:|---:|---:|---:|
| 2026 | P0 | 0.230515 | −0.008670 | 0.074489 | −0.061008 |
| 2026 | PVALID | 0.288094 | 0.012480 | 0.132165 | −0.019078 |
| 2026 | PLOCAL | 0.435270 | 0.082094 | 0.267593 | 0.018613 |
| 2026 | PLOCAL+GLOBAL | 0.447439 | 0.120877 | 0.272526 | 0.013734 |
| 2026 | PORACLE | 0.505433 | −0.061606 | 0.320742 | 0.051305 |
| 3026 | P0 | 0.220304 | −0.009900 | 0.069542 | −0.001839 |
| 3026 | PVALID | 0.228338 | −0.002590 | 0.100944 | −0.005022 |
| 3026 | PLOCAL | 0.413428 | 0.014231 | 0.198186 | −0.005673 |
| 3026 | PLOCAL+GLOBAL | 0.428225 | 0.055314 | 0.241582 | 0.005292 |
| 3026 | PORACLE | 0.504087 | 0.094769 | 0.250846 | 0.006127 |

PORACLE 对 rank information 的恢复跨 seed 很强，但 static natural R² 在 seed 2026 仍为负，说明 Oracle 并未统一修好 calibration；Strong Oracle gate 预注册的 primary metric 是 static macro Spearman。相比之下，PLOCAL+GLOBAL 的 static natural R² 在两个 seed 均转为正，并且是 primary mask-free variants 中最高。

完整 endpoint、Pearson、RMSE、MAE、variance ratio、descriptive calibration 和 transformed-fold summaries 见 [Table 2](../metrics/table2_static_ftv.csv) 与 [Table 3](../metrics/table3_delta_ftv.csv)。

## Legacy deficit recovery 与方向性

N1 相对 L1 的 P0 static macro Spearman deficit 为 0.300770 / 0.258429。预注册 recovery 为 pooling gain 除以该 deficit：

| Seed | Pooling | N1 absolute gain | Recovery |
|---:|---|---:|---:|
| 2026 | PVALID | +0.057579 | 19.1% |
| 2026 | PLOCAL | +0.204755 | 68.1% |
| 2026 | PLOCAL+GLOBAL | +0.216924 | 72.1% |
| 2026 | PORACLE | +0.274919 | 91.4% |
| 3026 | PVALID | +0.008034 | 3.1% |
| 3026 | PLOCAL | +0.193124 | 74.7% |
| 3026 | PLOCAL+GLOBAL | +0.207921 | 80.5% |
| 3026 | PORACLE | +0.283783 | 109.8% |

N3 对 matched L3 的 replication 方向一致：PLOCAL gain +0.187424 / +0.168882，PLOCAL+GLOBAL +0.198096 / +0.171312，PORACLE +0.255301 / +0.264969。完整结果见 [Table 4](../metrics/table4_legacy_deficit_recovery.csv)。

方向性不是只由 pooled endpoint 决定：N1 的 PLOCAL、PLOCAL+GLOBAL、PORACLE 相对 P0，在 static 和 ΔFTV 的两个 seed 中均为 10/10 fold-macro 正向。Static endpoint-fold 正向数分别为 35/40、36/40、39/40；ΔFTV endpoint-fold 为 24/30、26/30、26/30。PVALID 较弱：static 为 7/10 fold-macro、26/40 endpoint-fold，ΔFTV 为 8/10、19/30。

PLOCAL+GLOBAL 相对单独 PLOCAL 的 pooled static ρ 额外提高 +0.012169 / +0.014797，pooled ΔFTV ρ 额外提高 +0.004933 / +0.043396；N1 fold-macro 方向分别为 static 7/10、ΔFTV 8/10。N3 的 PLOCAL、PLOCAL+GLOBAL、PORACLE 相对 P0 仍各为 static/ΔFTV 10/10 fold-macro 正向，但 PLOCAL+GLOBAL 相对 PLOCAL 只在 static 5/10、ΔFTV 5/10 folds 正向。因此 global context 提供了小而方向偏正的 pooled 增量，主要恢复来自固定 local readout，而 concat 本身尚未显示逐 fold 稳定优势。

预注册 secondary `PLOCAL+PVALID` 只作 sensitivity：N1 static ρ 为 0.454413 / 0.422317，ΔFTV ρ 为 0.265905 / 0.236874；相对 primary PLOCAL+GLOBAL 分别是一升一降、两 seed 略降。N3 static 为 0.472351 / 0.431084，ΔFTV 为 0.291310 / 0.264698。它没有产生可替代 primary Local–Global 结论的一致方向。

## Padding 与 acquisition nuisance

按 seed 2026 / 3026，N1 P0 对 padding fraction 的 macro natural R² 为 0.191907 / 0.190527；valid-source fraction 在各 seed 中为相同数值，因为两者互为确定性补量。PVALID 没有降低它们，反而升至 0.224102 / 0.260348；同时 PVALID 的 FTV recovery 只有 19.1% / 3.1%。因此预注册 Padding / Geometry gate 为 `NOT_SUPPORTED_IN_PILOT`。

C1B P0 的确携带一些 acquisition geometry：N1 对 in-plane acquisition FOV 的 R² 为 0.312 / 0.317，对 max resample factor 为 0.123 / 0.110。PLOCAL 将这些值分别降至 0.116 / 0.145 与 0.031 / 0.026，同时大幅提高 FTV rank information；这与远端 context/geometry dilution 相容。不过 PLOCAL 对 padding/valid-source R² 基本不降（0.198 / 0.193），而 PVALID 的 nuisance decodability 上升，所以不能把主要机制写成 padding removal。

Legacy PVALID 为 `NA_no_source_authoritative_mask`；legacy PORACLE 为 `NA_incomplete_source_authoritative_support_1488_of_1500`。本轮没有把 lesion channel 当 valid-source mask，没有删除 12 visits，也没有用 zero/P0 fallback 伪造 legacy Oracle。完整 nuisance 结果见 [Table 5](../metrics/table5_nuisance_decodability.csv)。

## Occupancy、downsampling 与 training budget

固定旧 P0 OOF predictions 的 diagnostic 显示，occupancy 与 `N1 absolute error − L1 absolute error` 的 Spearman 在 seed 2026 的 T0–T3 为 −0.041、−0.075、+0.033、−0.003，在 seed 3026 为 −0.006、−0.066、+0.040、−0.005。效应接近零、方向不统一；endpoint 内 quartile 样本也高度不均，例如 T0-Q1 只有 1 个 observation、T3-Q4 只有 13 个。因此没有支持“occupancy 越低，C1B degradation 单调越严重”的完整证据。

Max-resample-factor 与 paired error difference 的 Spearman 在两个 seed、四 endpoint 的范围为 −0.108 至 +0.020；`>2` stratum 每 endpoint 只有 8–10 个 observations，误差方向不一致。Downsampling 不能解释主要 degradation。完整诊断见 [Table 6](../metrics/table6_occupancy_downsampling.csv)。

40/40 training cells 均未 hit configured epoch 12，40/40 selected checkpoint 都不在最后两个 observed epochs；各 arm selected epoch median 均为 3。N1/N3 last-three normalized validation-state-loss slope median 为 +0.220 / +0.197，表示 selected epoch 后并非仍在稳定下降。预注册 `UNDERTRAINING_PLAUSIBLE` 条件对 N1、N3 均为 false。见 [Table 7](../metrics/table7_training_budget.csv) 与 [training summary](../metrics/training_budget_summary.json)。

## 对 14 个问题的逐条回答

### 1. Existing P0 是否精确复现正式 Stage-B 结果？

是。40 个 cell、24,821,760 个 representation 元素全部 bitwise equal；probe selection、OOF prediction、pooled natural metrics 也精确一致。两个硬 STOP gates 均为 PASS。

### 2. C1B final spatial feature map 中是否仍存在 FTV signal？

是，而且是 substantial rank information。N1 PORACLE static ρ 从 0.230515 / 0.220304 升至 0.505433 / 0.504087；固定 PLOCAL 也升至 0.435270 / 0.413428。信息不是从 final spatial map 中消失，而是在现有 global aggregation 中大幅受损。

### 3. PORACLE 相对 P0 改善多少？

N1 static macro ρ 改善 +0.274919 / +0.283783；ΔFTV macro ρ 改善 +0.246253 / +0.181304。Static fold-macro 为 10/10 正向，endpoint-fold 为 39/40 正向。

### 4. PORACLE 能恢复多少 legacy deficit？

按预注册 primary recovery definition，恢复 91.4% / 109.8%。Seed 3026 的 PORACLE N1 static ρ 还高于同 seed L1 P0；这证明 spatial information exists，但不把 Oracle 变成部署方法。Natural R² 未在两 seed 同样恢复，仍须保留 calibration 限制。

### 5. Fixed 64-mm PLOCAL 是否有效？

是。N1 static ρ 提高 +0.204755 / +0.193124，恢复 68.1% / 74.7% deficit；ΔFTV ρ 提高 +0.193104 / +0.128644。Static 与 ΔFTV 均为 10/10 fold-macro 正向。窗口是 frozen central physical window，不使用 lesion、FTV 或 outcome recentering。

### 6. PLOCAL+GLOBAL 是否优于单独 PLOCAL / GAP？

明确优于 GAP；相对 PLOCAL 则是小幅、非逐 fold 一致的增益。Static ρ 额外 +0.012169 / +0.014797，static natural R² 额外 +0.038783 / +0.041083；ΔFTV ρ 额外 +0.004933 / +0.043396。N1 static/ΔFTV fold-macro 分别 7/10、8/10 优于 PLOCAL，N3 则均仅 5/10；因此它适合作为下一 pilot，但不应声称 concat 已稳健胜出。

### 7. PVALID 是否证明 padding 是主要问题？

否。PVALID static gain 只有 +0.057579 / +0.008034，未通过两 seed gate；padding/valid-source R² 不降反升。Padding 可能存在于 representation 中，但 valid-source-only aggregation 不是主要修复。

### 8. C1B representation 是否高度编码 padding/source geometry？

部分是。N1 P0 的 padding/valid-source R² 约 0.19，in-plane FOV R² 约 0.31，说明 geometry 可解码；相较 legacy，C1B 的 padding decodability 更高。但证据不支持它是 primary bottleneck：PVALID 增强而非削弱 nuisance，PLOCAL 改善 FTV 时主要降低的是 FOV/resampling decodability，而非 padding decodability。

### 9. Lesion occupancy 越低时 C1B degradation 是否更严重？

没有一致证据。Occupancy–paired-error correlations 在 −0.075 至 +0.040，接近零且跨 endpoint 变号；部分低-occupancy strata error 较大，但 endpoint 内极端 quartile 样本过小、rank 方向不单调。该 diagnostic 不能承担因果解释。

### 10. Downsampling 是否能解释主要 degradation？

否。相关范围只有 −0.108 至 +0.020，`>2` stratum 仅 8–10 observations/endpoint，方向也不一致。结果不支持把 global degradation 主要归因于 extreme resampling。

### 11. N arms 是否存在明显 undertraining？

否。N1/N3 均为 0% hit max epoch、0% selected in last two observed epochs，selected epoch median 3；selected 后 validation loss slope 为正而非继续下降。Training budget 不是本轮主要解释。

### 12. Bottleneck 主要位于 pooling、padding/geometry、encoder 还是 mixed？

唯一选择是 **A — POOLING BOTTLENECK**。Oracle strong、mask-free local recovery strong、padding gate false；因此不是 C，也不需要用 B 或 D 覆盖这条预注册层级结论。现有 encoder 已形成可读的 lesion-related spatial evidence，主要损失发生在 global spatial aggregation。

### 13. 是否需要执行 secondary S3 audit？

不需要，也不允许。Final PORACLE 两 seed 均通过 Strong Oracle Recovery gate，trigger 为 `NOT_TRIGGERED_FINAL_ORACLE_STRONG`。S3 保持 0 assets，避免事后扩展分析。

### 14. 下一步唯一优先级是什么？

唯一优先级是 **Local–Global Response State Pilot**：保留 C1B observability，以固定 64-mm local state 提供 response focus，并同时保留 global context state。该 pilot 应保持 mask-free；不回退 legacy fixed-voxel crop，不使用 PORACLE 作为 inference input，也不在本 audit 中训练。

## A–D 决策与边界

| Outcome | 状态 | 冻结证据映射 |
|---|---|---|
| A — Pooling bottleneck | SELECTED | PORACLE 两 seed strong；PLOCAL 与 PLOCAL+GLOBAL 两 seed均通过 deployable recovery gate。 |
| B — Padding / geometry dilution | NOT SELECTED | PVALID FTV recovery 未过门，padding/valid-source decodability 未下降。 |
| C — Encoder bottleneck | NOT SELECTED | PORACLE 远高于 P0，且 mask-free PLOCAL substantial recovery。 |
| D — Mixed bottleneck | NOT SELECTED | 冻结层级中 strong Oracle + strong deployable local + padding gate false 已唯一映射到 A。 |

本结论不表示现有 encoder 已达到最终性能，也不否认 future pretrained encoder 可能继续改善；它只确定下一实验的第一优先级。两个 training seeds 是 pilot-level 独立单位，fold、patient、visit 和 endpoint 不是独立重复；报告没有用 fold 数量制造显著性。

## 表与图

- Tables：[Table 1](../metrics/table1_feature_map_contract.csv)；[Table 2](../metrics/table2_static_ftv.csv)；[Table 3](../metrics/table3_delta_ftv.csv)；[Table 4](../metrics/table4_legacy_deficit_recovery.csv)；[Table 5](../metrics/table5_nuisance_decodability.csv)；[Table 6](../metrics/table6_occupancy_downsampling.csv)；[Table 7](../metrics/table7_training_budget.csv)。
- Gates：[P0 state equivalence](../metrics/p0_equivalence_gate.json)；[P0 probe replication](../metrics/p0_probe_replication_gate.json)；[S3 trigger](../metrics/s3_trigger_authorization.json)；[prospective classification](../metrics/prospective_gates.json)。
- Figures：[Figure 1](../figures/01_feature_map_pooling_schematic.png)；[Figure 2](../figures/02_pooling_weight_illustration.png)；[Figure 3](../figures/03_static_macro_spearman.png)；[Figure 4](../figures/04_static_natural_r2.png)；[Figure 5](../figures/05_delta_macro_spearman.png)；[Figure 6](../figures/06_legacy_deficit_recovery.png)；[Figure 7](../figures/07_oracle_vs_local_recovery.png)；[Figure 8](../figures/08_padding_valid_source_decodability.png)；[Figure 9](../figures/09_ftv_vs_nuisance_information.png)；[Figure 10](../figures/10_occupancy_n1_l1_degradation.png)；[Figure 11](../figures/11_selected_epoch_training_budget.png)；[Figure 12](../figures/12_representative_activation_montage.png)。

Figure 2 只显示 aggregate pooling-support weights；PORACLE panel 是 non-deployable lesion-support diagnostic。Figure 12 是固定 seed 2026/N1/fold 0 encoder 对 outcome-free hash rule 选出的 16 名患者×4 visits 的归一化 aggregate activation，不显示或写出 patient identifier，也不用于选择 pooling。

## 最终限制与科学链条

本 audit 只改变 deterministic readout，不改变 input、crop、checkpoint、encoder、transition、grounding weights 或 Ridge contract。PORACLE 的 mask 只用于 diagnostic；PLOCAL/PLOCAL+GLOBAL 才是 mask-free evidence。P0 与旧正式结果的双重复现门排除了“新 extractor/probe 改写 baseline”的解释。

最终链条为：C1B-H 改善 lesion observability 与 grounding safety，但现有 GAP representation 的 FTV rank information 很弱；final spatial map 的 Oracle readout 恢复约全部 legacy deficit；固定 64-mm local readout无需 lesion mask即可恢复约 68%–75%，Local–Global 恢复约 72%–80%；PVALID、occupancy、downsampling 与 undertraining 均不足以替代该解释。因此当前最小、可证伪且唯一优先的下一实验是 Local–Global Response State Pilot。
