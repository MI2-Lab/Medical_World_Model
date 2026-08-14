# Non-FTV Phenotype Target Decodability Audit

## 1. 研究问题、估计目标与边界

本实验回答：冻结的 C1B-H DCE7 MRI spatial representation 能否编码临床上有用的
FTV、LD、SPH 与 BPE measurement；尤其在去除可由 FTV 线性解释的成分后，LD、SPH、
BPE 的 phenotype residual 是否仍可从图像表征解码，以及该信息位于 encoder spatial map、
response state 还是 JEPA projector 的哪一层。

本实验是 **target screening / bottleneck localization**。它只训练 outer-fold-isolated
linear Ridge probe，不更新 encoder、response projection、JEPA projector/transition、EMA
target network 或 checkpoint，不启动 FTV+LD 或任何 multi-target World Model training，
也不把 Oracle mask representation 当作可部署输入。

Primary estimand 是冻结 representation 对以下量的 held-out OOF decodability：

1. 当前访视的 raw measurement；
2. 在 FTV 条件均值之外的 transformed-space residual；
3. 相邻访视的 percent-change 及其 FTV-residual；
4. 同一 target 在 representation pipeline 不同位置的可访问性。

FTV 是 response-control target；non-FTV target 是 LD、SPH、BPE。所有结论都是本 cohort
内的 representation audit，不等同于外部验证、生物学独立性、因果机制或临床可部署性。

## 2. 冻结 provenance 与 Goal 6 证据边界

本分支为 `feature/nonftv-phenotype-decodability-audit`，冻结 parent 为
`7742d737d92ed153b5c721cd323528b0a127d5ef`（short SHA `7742d73`）。

Goal 6 不在该 parent 的祖先历史中；它是 sibling worktree/branch 的冻结外部证据：

- sibling HEAD：`f49cf17237a95e9f8b99ad5f13c73f90e1a94a28`；
- Goal 6 experiment commit：`01f2b3dad409347e31adf2dbae9082293fe86950`；
- frozen final-report SHA-256：
  `c9b1d5a6b4e7ea1cb9b8eb5002fcfe281b5fec36ae79fea7cf259356b69dfd0b`。

因此，本实验可以把以下 aggregate 结果作为预先存在的 downstream-relevance prior，但
不得把它们说成已合入当前 parent、已由本实验复现，或已在当前 375 人 probe protocol
中重新证明：

- Goal 6 primary 是 384 人（pCR+=113）的独立五折协议；它支持联合
  `N = LD + SPH + BPE` 在 `Clinical + FTV` 后的 pCR increment；
- 联合 `N_res` 在 outer-train-only FTV residualization 后仍有 pCR increment；
- family ablation 仅给出描述性顺序：LD strongest、SPH weaker、BPE weakest；Goal 6
  **没有**分别证明 `LD_res`、`SPH_res`、`BPE_res` 各自有独立 pCR increment；
- limited profile evidence 是 aggregate `N -> HER2` signal，不是任一单独 family 的
  image decodability，也不是机制证据；
- Goal 6 的 matched MRI reference 是另外的 375 人描述性 sensitivity，preprocessing 与
  head 不同，不能用来断言当前 representation 已经或尚未编码某个 target。

Candidate scorecard 必须把 clinical relevance 拆成 `joint_N_evidence` 与
`family_specific_evidence` 两列。LD/SPH/BPE 的后者只能标记为描述性 family localization，
不能写成三个 family 都已分别通过 residual pCR 检验。

## 3. 冻结数据、population、fold 与模型

### 3.1 Population 与 timing

- 输入：C1B-H、DCE7，四访视 `T0,T1,T2,T3`；T3 始终标记
  `late/pre-surgery`。
- Primary population：与 radiomics workbook 精确 ID 相交的、FTV-complete、四访 MRI
  完整的 **375 名患者 / 1,500 visits**。不得用 384 人 Goal 6 primary 或 808 人 feature
  asset 总体替换这一 estimand。
- Patient join 只接受冻结的 canonical six-digit trial ID exact match；禁止 fuzzy join、
  row-order join 或按 target availability refill。
- Split：冻结 seed-2026 MRI manifest 的五个 outer folds `0..4`。每折 train、validation、
  test 必须互斥且覆盖 375 人；每名 eligible patient 在每个 seed/arm/representation 下恰好
 进入一次 outer test。所有表征与 target 一律按 `patient_id` 对齐，绝不按 NPZ row index
  假定一致。
- Static raw target 在该 375 人 matched cohort 上应完整；若实际 audit 不一致则 fail closed，
  输出 target/fold/visit missing ledger，不做插补或换 cohort。

Workbook 固定为 `Multi-feature-MRI-NACT-Data.xlsx`、sheet `datawith4visits`，SHA-256
`f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc`。Fold manifest
SHA-256 为 `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。
源文件只读；若 hash、sheet、ID cardinality、visit coverage 或 fold coverage 漂移，正式
probe 不得运行。

### 3.2 Frozen checkpoint matrix

只使用以下 20 个 test-blind `selected.pt`：

- seed bases：`2026, 3026`；每折 effective seed 为 `seed_base + fold`；
- arms：`LOCAL0`（无直接 FTV grounding）与 `LOCAL3`（Direct FTV grounding）；
- outer folds：`0,1,2,3,4`。

每个 checkpoint 的路径、SHA-256、selection record、selected epoch、
`test_data_used=false` 与配套 feature metadata 必须在看 target 结果前写入 preregistration
lock。不得重选 epoch、重训 encoder、改用 GAP/LG、跨 arm 共享不匹配的 projector，或根据
本实验 target/test 表现换 checkpoint。

冻结的输入几何为 float32 `[4,7,112,176,160]`、XYZ spacing
`[0.9,0.9,2.0] mm`。LOCAL 使用固定、mask-free、lesion-centered
`64 x 64 x 64 mm` physical support；encoder 与 JEPA 全部 `eval()`，不启用 stochastic
augmentation。

独立 parity 复算还必须用 checkpoint 的 `response_projection(Z3)` 对 GPU-exported Z2 做
float32 CPU 检查。跨 CPU/GPU kernel 的冻结容差为 `rtol=1e-5, atol=5e-6`，且 global
maximum absolute difference `<=5e-6`；这不替代 source experiment 自身 hash binding 与
exporter metadata parity。该容差在任何 retained formal output 前、一次严格
`atol=1e-6` pre-lock 检查因两个 cell 的近零值失败后冻结；20 cells 实测 global maximum
absolute difference 为约 `2.03e-6--2.62e-6`，属于 float32 platform roundoff，不是
target-result-driven amendment。

## 4. 精确 target contract

不得重命名、替换或重新计算 workbook 的绝对 measurement：

| Family | T0 | T1 | T2 | T3 | 定义与单位 |
|---|---|---|---|---|---|
| FTV | `VOLUME_TUM_BLU_V10` | `VOLUME_TUM_BLU_V20` | `VOLUME_TUM_BLU_V30` | `VOLUME_TUM_BLU_V40` | I-SPY2 PE/SER criteria 下的 functional tumor volume，cc；response control |
| LD | `LD_T0` | `LD_T1` | `LD_T2` | `LD_T3` | imaging-reported longest tumor diameter；workbook 未声明单位 |
| SPH | `SPHERICITY_T0` | `SPHERICITY_T1` | `SPHERICITY_T2` | `SPHERICITY_T3` | 由 3-D FTV tumor mask 得到的 sphericity，无量纲 |
| BPE | `BPE_5slice_mean_T0` | `BPE_5slice_mean_T1` | `BPE_5slice_mean_T2` | `BPE_5slice_mean_T3` | 对侧乳腺中央连续五层 fibroglandular tissue 的 mean early enhancement；native mean-PE scale，绝对单位未声明 |

LD/BPE 在后期访视出现的零值是合法 observation，不得当作 missing、替换为 floor 或删除；
只有当它作为 percent-change 分母时，才按下述零分母规则判该 interval 不可用。

### 4.1 Static target

对每个 `t in {T0,T1,T2,T3}`，分别运行：

`Z_t -> FTV_t, LD_t, SPH_t, BPE_t`。

T0/T1/T2 是 early/mid gate timings；T3 只报告为 `late/pre-surgery`，不用于
early/mid gate。

### 4.2 Dynamic target：相邻 percent-change 新扩展

Goal 6 冻结的是相对 baseline 的 `T0->Tk` change；它并未预注册 `T1->T2` 或
`T2->T3` 相邻 target。本实验为定位 response dynamics，**预先新增**以下 adjacent
percent-change contract，而不声称它是 Goal 6 已运行的 target：

`Delta_pct(x; a->b) = 100 * (x_b - x_a) / abs(x_a)`，

其中 `(a,b)` 依次为 `T0->T1, T1->T2, T2->T3`。本计划中的 `DeltaFTV`、`DeltaLD`、
`DeltaSPH`、`DeltaBPE` 均专指这个相邻 percent-change，不指自然单位绝对差，也不指
baseline-referenced `T0->Tk`。

- 分母必须 finite 且严格非零；否则该 patient 仅对该 family/interval 不 eligible；
- 禁止 epsilon、clamp、floor、median imputation 或用 workbook materialized baseline-change
  列替代；
- numerator/denominator 都只能来自该 interval 的两个已观察访视；
- 每个结果必须报告 eligible `n` 与零分母/非有限排除数；paired representation comparison
  使用完全相同 eligible patient set。

Primary dynamic input 是 literal latent difference：

`DeltaZ_(a->b) = Z_b - Z_a`。

Secondary sensitivity 是按固定顺序
`[Z_a, Z_b, Z_b-Z_a]`；例如 T0->T1 明确为 `[z0,z1,z1-z0]`。
Secondary 不进入 Gate E，也不能推翻 primary literal-difference 结论。

## 5. Frozen representation locations

所有 representation 都来自同一 cell 的 frozen **online** pathway。不得混用 EMA target
stream；每个 visit 均保持原 float32 feature 后再在 probe 中 train-only scaling。

| ID | 精确定义 | 维度 | 角色 |
|---|---|---:|---|
| Z1 | `projector(r_t)`，即 online JEPA `VisitProjector` 输出 | 192 | deployable；JEPA-projected state |
| Z2 | `r_t = response_projection(local_mean_t)`，即 projector 之前的 online response state | 192 | deployable；当前 response state |
| Z3 | final online encoder spatial map 在固定 64-mm LOCAL fractional weights 下的 weighted mean | 128 | deployable；raw mean-pooling reference |
| Z4 | `[LOCAL weighted mean, LOCAL weighted population SD]`，mean-first | 256 | deployable；Goal 5 P3 heterogeneity reference |
| Z5 | `[CORE weighted mean, CORE weighted population SD]` | 256 | mask Oracle，diagnostic only |
| Z6 | `[PERI10 weighted mean, PERI10 weighted population SD]` | 256 | mask Oracle，diagnostic only |
| Z7 | `[PERI20 weighted mean, PERI20 weighted population SD]` | 256 | mask Oracle，diagnostic only |

这里必须区分两个 projection：Z3 经 frozen `Linear(128,192)+LayerNorm(192)` 得到 Z2；
Z2 再经 JEPA nonlinear projector 得到 Z1。因此预注册的 **projection bottleneck** 比较是
`Z2 vs Z1`，不是把 Z2 错写成 raw 128-D LOCAL mean。

Z4--Z7 的 SD 使用与 Goal 5 一致的 weighted **population** variance。Oracle region
定义固定为：

- CORE：lesion voxels；
- PERI10：valid-source、non-lesion、到 lesion 距离 `(0,10] mm`；
- PERI20：valid-source、non-lesion、距离 `(10,20] mm`，不是累计 0--20 mm；
- 映射后 region occupancy 再乘与 Z3/Z4 完全相同的固定 LOCAL weight。

因此 Z5--Z7 没有扩大 FOV，仍受 64-mm LOCAL support 限制。它们依赖 lesion mask，永远
不能直接成为 deployment 或 grounding input 建议。

Spatial asset 中无效 Oracle row 是显式零，绝不能当作真实 feature。Static 必须要求当前
visit 的 `oracle_valid=true`；dynamic 必须同时要求 start/end 有效。冻结资产在 matched
cohort 的当前可用性基准为：CORE T0/T1/T2/T3 `375/375/374/374`，PERI10
`375/375/374/375`，PERI20 `375/375/375/375`；相邻有效数分别为 CORE
`375/374/373`、PERI10 `375/374/374`、PERI20 `375/375/375`。运行时必须重新核验这些
aggregate counts；不符即 fail closed。

若 Goal A mask-free regional representation 在本实验 lock 前已有 `COMPLETE`、hash-bound、
同 375 人/五折/两 seed/两 arm 的兼容资产，可登记为 Z8 secondary external diagnostic；
否则记录 `Z8_NOT_AVAILABLE_AT_LOCK` 并立即继续。本实验不得等待 Goal A，Z8 不进入任何
primary gate 或 recommendation rule。

## 6. Target preprocessing 与 residualization

所有步骤都在每个 outer fold 内重新拟合；test target 不能影响任何 boundary、均值、方差、
residualizer 或 alpha。

### 6.1 Raw target transform

对 static absolute target：

1. 仅用 outer train 拟合每列 1%/99% winsor boundary；
2. FTV、LD、BPE 在 clipping 后使用 `log1p`；SPH 使用 identity；
3. 仅用 outer train 的 transformed 值拟合 `StandardScaler`（population variance，
   `ddof=0`）；
4. 同一个 frozen transform 应用于 validation/test。

对 adjacent percent-change：先在 natural measurement 上按第 4.2 节计算，再以 outer-train
1%/99% winsor、identity transform、outer-train `StandardScaler` 处理。Signed change
不得使用 `log1p`。

### 6.2 Primary FTV residual

对每个 target、timing/interval、outer fold 分别拟合，不做全 cohort residualization：

- static：`LD*_t ~ FTV*_t`、`SPH*_t ~ FTV*_t`、`BPE*_t ~ FTV*_t`；
- dynamic：`DeltaLD* ~ DeltaFTV*`、`DeltaSPH* ~ DeltaFTV*`、
  `DeltaBPE* ~ DeltaFTV*`。

星号表示完成第 6.1 节 train-winsorized、family-transformed、standardized 后的值。模型是
带 intercept 的 sklearn Ridge，**固定 `alpha=1.0`**，只在 outer train 拟合；validation
与 test 只应用冻结映射。残差定义为：

`epsilon_y = y* - ridge_train(x*)`。

Primary dynamic residual 是对相邻 percent-change **直接 residualize**；不得改成两个
static endpoint residual 的差，也不得在 natural workbook units residualize。该 fixed-alpha
residualizer 与 Goal 6 中另一个 validation-tuned `NONFTV -> FTV` redundancy ridge 是不同
分析；本实验不运行、也不借用后者的 alpha selection。

### 6.3 Secondary stricter FTV+LD residual

仅对 SPH 与 BPE，增加：

- static：`SPH* ~ [FTV*,LD*]`、`BPE* ~ [FTV*,LD*]`；
- dynamic：`DeltaSPH* ~ [DeltaFTV*,DeltaLD*]`、
  `DeltaBPE* ~ [DeltaFTV*,DeltaLD*]`。

同样固定 Ridge `alpha=1.0`、outer-train only。它检验 volume + diameter 之外的信息，
必须标记 `secondary_ftv_ld_residual`，不进入 Gate B--E、primary target classification 或
正式 recommendation；不得在看到 FTV-only residual 结果后选择是否运行。

## 7. Ridge probe、fold isolation 与 OOF aggregation

每个 `(seed, arm, representation, target variant, timing/interval, input variant, fold)` 独立
运行：

1. representation 的 `StandardScaler` 只在 outer train 拟合；constant columns 用确定性
   zero-scaled 行为并记录；
2. target 使用第 6 节 frozen transform；对 transformed residual，probe outcome scaler
   也只在 outer-train residual 上拟合；
3. Ridge alpha grid 固定为
   `{1e-4,1e-3,1e-2,1e-1,1,10,100,1000}`；
4. 用 validation transformed-standardized MSE 最小值选 alpha；完全相同时取更小 alpha；
5. 选择后不在 train+validation refit；outer test 只调用一次 prediction；
6. 任一 non-finite、重复 ID、split overlap、target/feature patient-set mismatch、constant
   target 或 metric undefined 均 fail closed，并把该 cell 记为 invalid，不能以另一 fold、
   seed、representation 或 target 替补。

五个 untouched outer-test partition 合并成每个 seed/arm/representation/endpoint 的一组
patient-level OOF prediction，再计算 raw target 的 natural-space endpoint metric。不同 fold
的 transformed/residual 坐标不可直接拼接；所有 transformed/residual metric（包括 rank、
R2、RMSE/MAE、variance ratio 与 calibration）均在各 fold 各自坐标计算后按 test `n` 加权
汇总，同时保留五折值。Macro 是预注册 endpoints 的无权均值，不能用它隐藏任一
interval/timing。

所有跨 representation effect 必须在完全相同 patient intersection 上 paired 计算。可附加
2,000 次、按 outer-test fold 分层的 paired patient bootstrap 95% CI；CI 只作不确定性描述，
不新增或修改 gate。

## 8. Metrics 与尺度解释

每个有效 cell 至少报告：

- Spearman rho、Pearson r；
- natural R2 与 transformed R2；
- natural RMSE、MAE；
- `Var(prediction,ddof=0) / Var(target,ddof=0)`；
- conventional descriptive calibration slope（observed target 对 prediction 的回归）
  `Cov(target,prediction,ddof=0) / Var(prediction,ddof=0)`，理想值为 1，并同时报告 intercept；
- `n`、各 fold `n`、missing/zero-denominator/Oracle-invalid exclusions、selected alpha。

Raw target 的 natural prediction 由 fold-fitted scaler 与 family transform 反变换，并对原始
natural target 评分；winsorization boundary 本身不可“反变换”，因此必须同时公开原始尺度
与 transformed-space 结果，不能只报 rank。

Residual 本身位于 transformed standardized space，没有不依赖 conditional baseline 的唯一
natural-unit inverse。因此：

- beyond-FTV primary rank/R2 是 `epsilon_y` 对 probe prediction 的
  `residual_spearman` 与 `residual_transformed_r2`；
- natural 指标先把 `ridge_train(x*) + predicted_epsilon` 合成完整 target prediction，再按
  fold transform 反变换，并对原始 target 计算，字段名固定为
  `reconstructed_natural_r2/rmse/mae`；
- 该 reconstructed metric 是 FTV conditional baseline + MRI residual readout 的整体重建
  guardrail，不得包装成“自然单位 residual R2”。

Primary ranking 必须联合阅读 Spearman 与 natural R2；高 rank 但 R2、variance ratio 或
calibration 严重异常时不能称为充分解码。

## 9. 必须运行的矩阵

### 9.1 Static matrix

对 Z1--Z7、两个 arms、两个 seeds、四个 timings，完整生成：

- raw `FTV/LD/SPH/BPE`；
- primary `LD_res_FTV/SPH_res_FTV/BPE_res_FTV`；
- secondary `SPH_res_FTV_LD/BPE_res_FTV_LD`。

### 9.2 Dynamic matrix

对三个 adjacent intervals 完整生成：

- primary input `Z_b-Z_a` 对 raw `DeltaFTV/DeltaLD/DeltaSPH/DeltaBPE`；
- primary input对 `DeltaLD_res_FTV/DeltaSPH_res_FTV/DeltaBPE_res_FTV`；
- secondary prefix input `[Z_a,Z_b,Z_b-Z_a]` 的同一 raw/primary-residual targets；
- FTV+LD stricter residual 只作为 secondary matrix。

Primary dynamic gate 只读 literal difference + FTV-only residual rows。所有表必须显式包含
`target_definition=adjacent_percent_change_new_extension`，防止与 Goal 6 baseline-reference
结果混表。

## 10. Primary gates 与 deterministic selection

Deployable candidate 定义为固定 `(arm, Z)`，其中 arm 为 LOCAL0/LOCAL3，Z 为 Z1--Z4；
Z5--Z8 永远不进入 deployable gate。一个 gate 要求“两个 seeds”时，必须是同一 arm、同一
Z、同一 target、同一 timing/interval，不允许 seed 2026 取一个 representation、seed 3026
取另一个，也不允许用 seed 均值掩盖单 seed 失败。等号只在写有 `>=` 的 gate 中通过；
undefined metric 一律失败。

为报告“最佳 deployable representation”而不手工挑 cell，按以下顺序确定同一个 candidate：

1. 最大化满足该 target gate 子条件的 early/mid timing 数；
2. 再最大化 early/mid timings 上的 minimum-over-seeds Spearman macro；
3. 再最大化 minimum-over-seeds natural R2 macro；
4. 完全相同则按 `Z1,Z2,Z3,Z4`，再按 `LOCAL0,LOCAL3` 的注册顺序取先者。

这是 phenotype decodability audit 内的预注册结果汇总，不是 model training selection；必须
同时展示全部 candidate matrix 和 multiplicity caveat。

### Gate A — `LD_IMAGE_OBSERVABLE`

存在一个固定 deployable candidate，在 `T0,T1,T2` 至少两个 timing 上，两个 seeds 均满足：

- raw LD Spearman `>= 0.40`；
- raw LD natural R2 `> 0`。

### Gate B — `LD_BEYOND_FTV_DECODABLE`

存在一个固定 deployable candidate，在 `T0,T1,T2` 至少一个 timing 上，两个 seeds 的
`LD_res_FTV` Spearman 均 `> 0.20`，且方向一致。

原 brief 的“natural R2 不系统性严重为负”预先操作化为：qualifying cell 的两个 seeds
不得同时 `reconstructed_natural_r2 < 0` 且两 seed mean `<= -0.25`。该数值仅是防止严重
幅度失配的 gate guardrail，不是显著性阈值。

### Gate C — `SPH_BEYOND_FTV_DECODABLE`

存在一个固定 deployable candidate，在 `T0,T1,T2` 至少一个 timing 上，两个 seeds 的
`SPH_res_FTV` Spearman 均 `> 0.20` 且为同一正方向。

### Gate D — `BPE_BEYOND_FTV_DECODABLE`

存在一个固定 deployable candidate，在 `T0,T1,T2` 至少一个 timing 上，两个 seeds 的
`BPE_res_FTV` Spearman 均 `> 0.20` 且为同一正方向。即便通过，也必须先通过第 12 节 FOV
observability audit，才可讨论 BPE grounding。

### Gate E — `NONFTV_DYNAMIC_SIGNAL_SUPPORTED`

对 `DeltaLD_res_FTV`、`DeltaSPH_res_FTV`、`DeltaBPE_res_FTV` 中至少一个 target，存在
一个固定 deployable candidate 和 early interval `T0->T1` 或 `T1->T2`，使 literal
`Z_b-Z_a` probe 的两个 seeds Spearman 均 `> 0.20`。`T2->T3`、prefix input 与
FTV+LD residual 均不授权 Gate E。

## 11. Bottleneck diagnostics

所有 effect 都是同 arm、target variant、timing/interval、eligible patient set 的 paired OOF
Spearman difference，并分别要求两个 seeds；不得跨 arm 或跨 population 相减。

### 11.1 Projection

若对任一 primary FTV-residual target，`Z2 - Z1 >= +0.10` 在两个 seeds 同时成立，标记
`RESPONSE_PROJECTION_DISCARDS_NONFTV_INFORMATION`，并记录 target/timing/arm。Raw-only
改善另行标为 `RAW_RESPONSE_PROJECTION_EFFECT`，不冒充 beyond-FTV evidence。

### 11.2 First-order pooling / statistic

若 `Z4 - Z3 >= +0.10` 在同一 target cell 的两个 seeds 成立，标记
`MEAN_POOLING_DISCARDS_TARGET_INFORMATION`。Residual target 的通过可参与 bottleneck
解释；raw-only 通过只作 burden/appearance diagnostic。

### 11.3 Oracle localization

为控制统计类型，primary localization contrast 是每个 Z5/Z6/Z7 的 mean+SD 相对 Z4
Full-LOCAL mean+SD，而不是利用维度/SD 差异与 Z3 作主比较。若至少一个 Oracle region
在同一 cell 两个 seeds 均满足 `Oracle - Z4 >= +0.10`，标记
`TARGET_SPATIALLY_LOCALIZED`，并明确 region、raw/residual target 与 timing。相对 Z3 的
结果完整报告但只作 secondary。

Oracle 不能成为训练输入建议。只有 residual target 的 localization evidence 才能支持
“先做 mask-free region-aware architecture 再 grounding”；raw-only localization 不能证明
FTV 之外的 phenotype 已定位。

### 11.4 Per-target bottleneck mapping

最终报告可同时列出多个机制 flag，但主要 bottleneck 按以下 fail-closed 层级解释：

1. BPE source 不在输入 support：`input_observability`；
2. Oracle residual 强而 Z4 弱：`localization`；
3. Z4 residual 强而 Z3 弱：`pooling/statistic`；
4. Z2 residual 强而 Z1 弱：`projection`；
5. FOV 可观察、Z1--Z7 全部未过 residual threshold，且无上述 effect：`encoder/current_feature_map`；
6. 证据组合不满足唯一归因：`mixed_or_unresolved`，不得强行写 encoder failure。

## 12. BPE input-target observability audit

BPE 的冻结 source definition 是**对侧乳腺**中央五层 fibroglandular tissue early
enhancement，而 Z1--Z7 均源自 lesion-centered 64-mm LOCAL support；Z5--Z7 也没有扩展
这个 support。因此在读 decodability 结果前，BPE 的 a-priori 状态为
`FOV_MISMATCH_SUSPECTED`。

正式 observability audit 必须只读地记录：target source definition 证据、C1B acquisition/
crop side 与 physical extent、LOCAL support physical extent、是否存在可 hash-bound 的 BPE
source ROI/coordinate mapping，以及 source ROI 与 model/LOCAL support 的 overlap。若能得到
source ROI，预先以 `>=99%` source occupancy 被 support 覆盖且无 support-boundary touch 定义
`OBSERVABLE_IN_FOV`；`<99%` 或明确 contralateral-vs-ipsilateral disjoint 定义
`TARGET_NOT_OBSERVABLE_IN_LOCAL_FOV`。若 source ROI/side mapping 不可得，则记录
`FOV_OBSERVABILITY_UNVERIFIED`，不得从低 Spearman 推断 encoder failure。

若 BPE 在 Full LOCAL/CORE/PERI 全弱，首先报告上述 FOV 状态。只要状态不是
`OBSERVABLE_IN_FOV`，BPE 就不能进入 grounding；Oracle-local failure也不能补偿缺失的
contralateral anatomy。

## 13. Scientific classification 与 recommendation

每个 LD/SPH/BPE 分别给出 categorical scorecard，不计算手工权重总分：

1. clinical relevance：Goal 6 joint-N evidence + family-specific descriptive strength；
2. image observability：raw gate/完整矩阵；
3. beyond-FTV：primary FTV-residual gate；
4. longitudinal organization：literal-difference dynamic gate；
5. stability：两 seeds 是否在同一固定 candidate 上方向一致；
6. input observability：尤其 BPE FOV status；
7. bottleneck：projection/pooling/localization/encoder/mixed。

原 brief 只为 LD 给出了 raw-target Gate A，没有为 SPH/BPE 指定 Class A/B 所需的 raw
threshold。为避免看结果后凭描述挑选，本 lock 前统一冻结一个**仅用于分类、不新增 primary
gate** 的 `RAW_IMAGE_SUPPORT_FOR_CLASSIFICATION`：对每个 family，存在同一个固定 deployable
candidate，在 T0/T1/T2 至少两个 timing 上两个 seeds 均满足 raw Spearman `>=0.40` 且 raw
natural R2 `>0`。LD 的该字段与 Gate A 同义；SPH/BPE 只用于 Class A/B categorical rule，
必须同时展示完整 raw matrix 并注明这是 brief 未指定阈值后的 pre-result operationalization。

分类规则：

- **Class A — STRONG GROUNDING CANDIDATE**：有 Goal 6 relevance prior，raw 与 FTV-residual
  均过相应 gate，至少一个 early dynamic residual cell 过 Gate E，且两 seeds 稳定、FOV
  compatible。它仍不等于该 family 已被 Goal 6 单独证明有 residual pCR increment。
- **Class B — RESPONSE-ONLY CANDIDATE**：raw 可解码但 residual gate 失败；解释为 burden/
  extent proxy，不称为 non-FTV phenotype-compatible target。
- **Class C — SPATIALLY LOCALIZED TARGET**：deployable residual 弱，但 Oracle residual
  localization diagnostic 通过；先解决 mask-free regional representation。
- **Class D — CURRENTLY NOT IMAGE-OBSERVABLE**：FOV 已确认可观察，但 deployable 与 Oracle
  均弱且无 projection/pooling/localization evidence。BPE FOV 未确认时禁止赋 D。

LD 即使是 Goal 6 描述性最强 family，也只有 Gate B 通过时才可称
`non-FTV phenotype-compatible grounding candidate`；否则只能称 morphology/extent
response target。

最终报告必须按以下优先级输出**唯一一个**主 recommendation，不自动启动 training：

1. `FTV + LD`：LD 为 Class A；
2. `FTV + regional LD/SPH`：没有 Class-A LD，但 LD 或 SPH 的 residual Oracle localization
   通过；这里“regional”指下一步开发 mask-free region-aware representation；
3. `Need broader-context phenotype branch`：前两项不成立，且 BPE 为
   `TARGET_NOT_OBSERVABLE_IN_LOCAL_FOV` 或 `FOV_OBSERVABILITY_UNVERIFIED`；
4. `FTV only; phenotype target not yet image-observable`：以上均不成立且 residual targets
   均弱。

## 14. pCR firewall、leakage 与禁止的 selection

本实验不得读取或使用 patient-level pCR 来选择 representation、target、residualizer、timing、
interval、probe alpha、gate 或 recommendation。Fold parser 只 allowlist ID/fold/split；即使
manifest 含 `label_pcr`，该列也不得进入 analysis object。不得生成 pCR prediction、AUROC、
AUPRC 或以 test pCR 作 post-hoc tie-break。

pCR relevance 只允许来自第 2 节 SHA-locked Goal 6 **aggregate** report/tables，并原样保留
n=384、joint-family 与 sibling-evidence caveat。当前 375 人的 target OOF test metric 可用于
本审计预注册 gate，因为它正是 image-observability estimand，但不得回流重选 checkpoint、
改 gate 或开始训练。

所有 transform、X/Y scaler、residualizer 与 Ridge alpha selection 只读 outer train/
validation；outer test 完全 untouched 至单次 prediction。T3 不得被包装成 early actionable
evidence。

## 15. Privacy、artifact 与 reproducibility contract

Patient-level ID、target、feature、split、Oracle validity、OOF prediction 与 residual 必须：

- 存为 `*.private.*`，文件 mode `0600`、private directory mode `0700`；
- 位于 `.gitignore` 覆盖范围，运行前后用 `git check-ignore` 与 tracked-file scan 验证；
- 不出现在日志、异常消息、figure label、public CSV/JSON/Markdown 或 Git diff；
- 不复制 raw MRI、workbook、large checkpoint 或 sibling private NPZ 到实验目录。

空间特征只从 sibling worktree 中已完成的 private asset 原位只读加载；其 metadata
hard-bound 绝对路径时不得搬移后绕过 provenance validator。Z3--Z7 的 upstream raw spatial
map 未持久化，因此任何新 pooling 定义都必须停止并另行授权重新提取，不能从现有 aggregate
伪造。

公开/可提交内容仅限 code、config、hash/provenance、aggregate metrics、aggregate figures、
中文 report 与不含 ID 的 manifest。Public table 的一行至少是一个 aggregate analysis cell，
包含 `n`、排除数与 patient-set SHA-256，但不能包含 patient ID 或逐人 prediction。Report 与
figure generator 只读取 aggregate metrics；private OOF 不得在 rendering 阶段重新打开。

在看任何 target result 前生成 immutable lock，至少绑定：parent SHA、Goal 6 sibling commit/
report hash、workbook/sheet hash、fold manifest hash、eligible target-table hash、20 个 checkpoint/
selection/feature metadata hash、spatial completion/hash、plan/config/scripts hash，以及每 cell
patient-order hash。任一 drift、partial matrix、privacy failure 或 provenance mismatch 都使 run
状态为 `FAILED_CLOSED`，不得发布部分 gate。

## 16. 必须产物与 final report

公开 aggregate tables 至少包括：

- `static_target_matrix.csv`；
- `residual_target_matrix.csv`（FTV-only primary 与 FTV+LD secondary 明确分层）；
- `dynamic_target_matrix.csv`（literal difference 与 prefix secondary 明确分层）；
- `representation_location_comparison.csv`；
- `oracle_localization_comparison.csv`；
- `bpe_fov_observability_audit.csv`；
- `grounding_candidate_scorecard.csv`（无 weighted total）；
- `final_target_recommendation.csv`；
- fold/alpha/exclusion aggregate audit 与 machine-readable gate JSON。

`reports/final_report.md` 必须用中文逐项回答：

1. FTV decodability control 如何；
2. LD raw 是否稳定可解码；
3. LD 去除 FTV component 后是否仍可解码；
4. SPH raw 与 residual 是否可解码；
5. BPE raw 与 residual 是否可解码；
6. BPE 是否存在 input/FOV observability 问题；
7. 哪些 adjacent Delta target/residual 可由 literal longitudinal latent difference 解码；
8. Z2 `r` 是否优于 Z1 `projector(r)`；
9. Z4 mean+SD 是否优于 Z3 mean；
10. Oracle CORE/PERI10/PERI20 是否显著改善 target，并基于何种 validity cohort；
11. 每个 target 的主要 bottleneck 属于 encoder、pooling、projection、localization、
    input observability 还是 unresolved；
12. 哪些 target 值得进入下一轮 grounding；
13. `FTV+LD` 是否得到充分的 image-observability/beyond-FTV 支持；
14. SPH/BPE 是否应等待 region-aware 或 broader-context architecture；
15. 下一轮正式 training 必须继续冻结哪些 data/model/timing/privacy contracts。

报告还必须披露 branch、最终 commit SHA、push status；push 失败时保留 local commit，记录
真实错误并写 `GITHUB_PUSH_FAILED`。禁止 force push。提交前检查旧实验目录未被修改，不提交
raw MRI、workbook、patient-identifiable table、private feature/prediction 或 large checkpoint。

## 17. 执行顺序

1. 完成只读 input/target/BPE-FOV inventory 与所有 hash/coverage check；
2. 冻结 preregistration lock，之后不得改 target、formula、gate 或矩阵；
3. 生成/验证 Z1，并验证 Z2 与 Z3--Z7 的 checkpoint、P1 parity、ID、split 与 validity；
4. 运行完整 static、residual、dynamic probe matrix；
5. 在完整矩阵通过 schema/OOF/privacy validation 后一次性计算 aggregate metrics、effects、
   gates、classification 与 recommendation；
6. 仅由 aggregate artifacts 生成 figures 与中文 final report；
7. 运行复现实验验证、Git privacy scan、旧目录 diff check，再按交付 contract commit/push。

任何中途结果都不得授权修改剩余实验，也不得自动开始 multi-target training。
