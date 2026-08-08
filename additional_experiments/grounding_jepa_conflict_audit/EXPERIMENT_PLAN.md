# Grounding–JEPA Conflict Audit 实验计划

## 1. 目标与边界

本实验只诊断现有 G3 中 Direct static FTV grounding 与 JEPA base objective 的联合优化关系。核心问题是：25 个 seed×fold G3 run 中出现的 validation base-loss degradation，是否与 shared image representation 上的两种梯度冲突有关。

本轮严格禁止：

- 重新训练 G1 或 G3；
- 修改架构、`lambda_FTV=0.25`、transition、biomarker 或 checkpoint selection；
- 实现或试跑 PCGrad、warm-up、two-stage、adaptive lambda、checkpoint averaging；
- 使用 pCR、treatment、clinical、subtype 或其他监督信号；
- 修改 `direct_grounded_response_state/` 或 `g3_multiseed_generalization/`。

所有新增文件只写入 `additional_experiments/grounding_jepa_conflict_audit/`。

## 2. 冻结输入

正式网格固定为：

- training seed base：2026、3026、4026、5026、6026；
- outer fold：0–4；
- 模型：仅 G3；
- selected checkpoint：每格 `best.pt`；
- 配对既有结果：同 seed、同 fold 的 G1/G3 validation state loss 与 FTV probe 指标；
- 总计 25 个 G3 selected checkpoints，不读取 smoke run 作为正式结果。

正式开始前必须对以下对象做 SHA-256 闭环：既有实验计划、训练实现、model/data/target 源码、25 个 G3 checkpoint、25 个 G3 history、25 个 selection、五个 FTV transform、final stability 与 seed-fold effect 表。任何一项漂移都停止正式分析。

本实验自身先生成不可覆盖的 `PLAN_FREEZE.json`，绑定本计划、`audit.yaml` 与 gradient/batch 核心实现。Phase A 后按不读取 gradient 的固定 degradation rank 选定 3 PASS/3 FAIL，并由 `SOURCE_CONTRACT.json` 进一步绑定 25 个 selected、6 个 representative last、公开/私有 batch commitment 以及实际输入的 964 个 NPZ 内容。外部 cache 的逐文件 patient 映射与 HMAC key 仅保存在 ignored private 目录；公开合同只保存无患者标识的 aggregate hash 与 keyed HMAC。正式 runner 必须 fail-closed 核验这些冻结合同。

## 3. 真实 computational graph

代码确认的 G3 forward 为：

```text
DCE7 -> SpatialVisitEncoder3D -> GAP
     -> response_projection(128 -> 192 + LayerNorm) -> r

r -> VisitProjector -> causal transition -> normalized next-state state_loss
  -> SIGReg(online projected state)
  -> L_base = state_loss + 0.09 * SIGReg

r -> Linear(192, 1) -> patient-mean SmoothL1 FTV loss

L_total = L_base + 0.25 * L_FTV
```

两种 raw loss 梯度的真实参数交集只有：

- `encoder.*`：28 个参数张量，892,672 个参数；
- `response_projection.*`：4 个参数张量，25,152 个参数；
- `all_shared`：32 个参数张量，917,824 个参数。

`projector.*` 与 `transition.*` 是 base-only；`ftv_head.*` 是 FTV-only；所有 `target_*` 是冻结 EMA 分支。它们不得进入 shared-gradient cosine。

## 4. Phase A：既有 run-level 指标

在任何新 forward/backward 之前，机械合并现有表和 selected history，生成 `metrics/run_level_existing_metrics.csv`，每个 seed×fold 恰一行。至少包含：

- seed、fold、effective seed、selected epoch；
- G1/G3 validation state loss、base degradation、base gate；
- static FTV ΔSpearman、observed ΔFTV ΔSpearman 与 ΔR²；
- selected representation std；
- selected train total/base/FTV loss 与 validation base-objective/state/FTV loss；
- last epoch、early-stop gap、逐 epoch grounded exposure 汇总。

先计算 base degradation 与 static gain、observed ΔFTV Spearman gain、observed ΔFTV R² gain 的 Spearman rho、双侧 crossed-exact permutation p、Holm p 和 20,000 次 crossed seed×fold bootstrap 95% CI。三个端点构成固定 family；n=25，仅作趋势诊断。

## 5. 固定 audit batch

### 5.1 样本池

- `train` 使用 formal G3 实际训练池：outer primary train 加 156 名无 FTV 的 I-SPY1 extra-pretrain 患者；
- `validation` 使用对应 outer validation 患者；
- 不读取或使用 pCR label；PatientRecord 中 pCR 固定为缺失值。

### 5.2 固定抽样

- `audit_seed=20260808`；
- 每 fold、每 split 8 个 batch；
- batch size=32；
- 每批至少 8 名有 FTV 患者；
- 按 `I-SPY2+FTV / I-SPY2无FTV / I-SPY1无FTV` 三个 stratum，依各 fold/split 原始比例用 largest-remainder 分配 32 个名额；
- 各 stratum 采用固定 seed 的 balanced-cycle permutation；batch 内禁止重复，跨 batch 只在样本池不足时均衡复用；
- 同 fold 的五个训练 seed 使用完全相同的 16 个 batch。

Validation 只有 121 人，因此 8×32 不可能全部互不重复；跨 batch 复用是预注册设计，推断仍以 25 个 run 为单位，绝不把 batch 当独立样本。

公开 `configs/audit_batch_manifest.csv` 每 batch 一行，只含计数、pool 统计与绑定 `batch_id/fold/split/index` 的整批有序成员 HMAC，不含患者 ID、逐患者 hash 或私有 roster 的普通 SHA。私有 membership、cache-input mapping 与 HMAC key 写入 ignored `configs/private/`，以 mode 0600 原子创建；公开私有映射 commitment 也只使用 keyed HMAC。

## 6. Gradient extraction

每个 selected G3 checkpoint 对固定 batch 执行两个完全独立的 forward/backward：

1. 清空梯度，固定该 batch 的 stochastic seed，计算并 backward `L_base`；
2. 再次清空梯度并重新 forward，计算并 backward raw `L_FTV`。

禁止由 `L_total.backward()` 反推分量，禁止创建 optimizer，禁止 `optimizer.step()`，并在每格前后验证所有参数值未改变。

主分析采用 `model.train()` 以保留原 transition dropout 的训练图；dropout 与 SIGReg 的随机流由 fold/split/batch 派生的、跨 training seed 相同的固定 seed 控制，因此每个 run/split 覆盖 8 个预先固定的 stochastic seeds，而不是单一 mask。两次 component forward 必须在同 seed 下逐 tensor bitwise 一致。固定启用 deterministic algorithms、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、cuDNN deterministic、关闭 benchmark/TF32，并使用 highest float32 matmul precision。EMA target 分支仍由模型 forward 强制 eval 且 detached。源训练记录为 `deterministic=false`；本设计是固定 checkpoint 上的 controlled training-graph post-hoc estimand，不声称 bit-exact replay 原训练 minibatch、batch order、dropout mask 或 SIGReg direction。

### 6.1 参数组

正式组固定为：

1. `encoder_overall`；
2. `encoder_stage_1 = encoder.features.0`；
3. `encoder_stage_2 = encoder.features.1`；
4. `encoder_stage_3 = encoder.features.2`；
5. `encoder_stage_4 = encoder.features.3`；
6. `response_projection`；
7. `all_shared = encoder + response_projection`。

若任何预期 shared parameter 对任一 loss 的 gradient 为 `None`，必须记录并令该组指标未定义；不得用零向量伪装。零 norm 同样记录为未定义并触发验收失败。

### 6.2 指标

对每个 run/split/batch/group 保存：

```text
cosine = (g_base · g_ftv) / (||g_base|| ||g_ftv||)
R = ||0.25 g_ftv|| / ||g_base||
M_base = (g_base · (g_base + 0.25 g_ftv)) / ||g_base||²
M_ftv = (g_ftv · (g_base + 0.25 g_ftv)) / (0.25 ||g_ftv||²)
```

并保存 raw dot、两种 norm、loss scalars、`cos<0`、`cos<-0.1`、`cos<-0.25` 与 `M_base<0` flags。

`M_base<0` 表示一阶近似下 joint update 直接增加 base objective；`0<M_base<1` 表示仍下降但被削弱；`M_base>1` 表示 FTV 帮助 base descent。

## 7. 聚合与统计单位

Batch 只用于同一个 run 内估计局部梯度分布。所有相关、PASS/FAIL 检验与 hypothesis 判定必须先把 8 个 batch 聚合为一个 seed×fold run 值；正式统计单位固定为 25 个 run。

每个 run、split、group 报告：mean/median cosine、负 cosine 比例、`cos<-0.1/-0.25` 比例、mean/median norm ratio、mean/median `M_base`、`M_base<0` 比例及 secondary `M_ftv`。H1/H2 的五个 core run endpoint 固定为 8 batch 的 median cosine、negative fraction、median norm ratio、median `M_base`、`M_base<0` fraction；8 batch 任一 core 值非有限即验收失败。

run 固定按 `seed_base, fold` 排序，未四舍五入的 `base_degradation=G3/G1−1` 在全流程只分类一次，必须机械断言 17 PASS/8 FAIL。

### 7.1 Correlation

对 train 与 validation 分开计算：

- base degradation vs median cosine；
- base degradation vs negative fraction；
- base degradation vs median norm ratio；
- base degradation vs median `M_base`；
- base degradation vs selected epoch / history / exposure 指标。

报告 Spearman rho、双侧 crossed-exact permutation p、预声明 family 内 Holm p 与 20,000 次 crossed seed×fold percentile bootstrap CI。每个 bootstrap replicate 分别有放回抽 5 个 seed levels 与5个 fold levels，再取二者 Cartesian product 的25格；同一 replicate 的索引同步用于所有 split/group/endpoint，重复格按 multiplicity 展开并重新 average-rank。相关检验穷举 `5!×5!=14,400` 个 seed-order×fold-order，把 outcome grid 的行列相对固定 degradation grid 重新标号；这保留 outcome 内的 seed/fold 结构，identity 包含在完整枚举内，双侧 exact `p=count(|T_perm|≥|T_obs|)/14,400`。bootstrap 索引与完整 permutation order 矩阵均保存 SHA、shape、dtype；不得因某个端点 nonfinite 而重抽。

### 7.2 PASS vs FAIL

使用既有阈值：base degradation≤5% 为 PASS，>5% 为 FAIL；预期 17 PASS / 8 FAIL。H1/H2 固定组效应为：

```text
D_cos   = median_FAIL(run median cosine) − median_PASS(...)
D_neg   = mean_FAIL(run negative fraction) − mean_PASS(...)
D_mbase = median_FAIL(run median M_base) − median_PASS(...)
D_mfail = mean_FAIL(run M_base<0 fraction) − mean_PASS(...)
D_ratio = median_FAIL(run median norm ratio) − median_PASS(...)
Q_ratio = median_FAIL(run median norm ratio) / median_PASS(...)
```

ratio 分母必须严格大于0。另报告每个 run 聚合值的：

- PASS 与 FAIL mean±SD、median/IQR；
- FAIL−PASS mean/median difference；
- `5!×5!=14,400` 个 crossed-exact permutations：固定原始17/8 gate grid，把 metric outcome grid 按完整 seed-order×fold-order 重标号，保留 outcome 的两向聚类结构；所有端点同步使用相同顺序，identity 已包含，difference 的双侧 exact `p=count(|T_perm|≥|T_obs|)/14,400`；`Q_ratio` 的 p 使用 `D_ratio` permutation；
- 20,000 次 crossed seed×fold bootstrap：对完整 run grid 的 `(metric, gate)` cell pair 同步做 seed-level 与 fold-level 有放回抽样，再在25格 multiset 内计算 PASS/FAIL mean、median difference 与 ratio；若某 replicate 缺任一组或 ratio 分母非正则记 nonfinite，不重抽。所有端点同步复用同一组索引；
- Mann–Whitney U 固定 two-sided asymptotic + continuity correction，只作敏感性分析。

描述 SD 使用 `ddof=1`，Q1/Q3 与 percentile CI 使用 NumPy linear quantile。禁止 BCa、studentization、winsorization、插补或重抽 invalid replicate；必须报告 requested/finite/nonfinite/fraction，finite fraction<0.95 时 CI 不可用。原始 core run nonfinite 令验收失败；常量相关输入记 `constant_input`、rho/p/CI 为 NA，Holm 中按 p=1，不是数据质量失败。普通95% CI只描述不确定性，不进入 H1–H3 gate。

所有判定 family 保留全部预声明成员，NA p 按1。Holm 按 raw p 升序、tie 按 endpoint id，使用 step-down cumulative maximum 并截断到1；同时报告 raw/Holm p、family id/size/status。固定 families 为：Phase-A gain 3项；H1 validation all-shared 6项；H2 validation all-shared 2项；H3 dynamics 8项；layer-localization validation 5层×3指标=15项。诊断门槛仅为 `p_holm≤0.10`，不得按结果缩小 family。

## 8. H1–H4 预注册判定

Primary split 为 validation；train 是相同 25 个模型上的 patient-composition sensitivity/replication，不是统计独立样本。所有 practical threshold 均在结果揭盲前固定。

### H1：真实方向冲突

五个信号：

1. FAIL−PASS median cosine ≤−0.10；
2. FAIL−PASS negative fraction ≥+0.10；
3. FAIL−PASS median `M_base` ≤−0.10；
4. degradation 与 cosine 或 `M_base` 的方向正确且 |rho|≥0.35；
5. FAIL−PASS `M_base<0` 比例 ≥+0.05。

其中第4项机械定义为 `rho(degradation, cosine)≤−0.35 OR rho(degradation,M_base)≤−0.35`。Validation 五项分别记 V1–V5。Train 同名方向信号为：`D_cos<0`、`D_neg>0`、`D_mbase<0`、至少一个令 V4 成立的同名 rho 在 train `<0`、`D_mfail>0`。当且仅当 validation `sum(V)≥3`、train 至少3项同方向，并且至少一个“已经越过对应 practical threshold 的 V endpoint”在 H1 六端点 family 中 `p_holm≤0.10`，判 H1。未越过 practical threshold 的显著端点不能代替证据条件。

### H2：gradient scale imbalance / curvature risk

仅在 H1 不成立且 H1 validation primary signal 数<3时判断。要求 validation 同时满足：

- FAIL/PASS median norm-ratio 比值≥1.50，或 FAIL−PASS≥0.25；
- degradation 与 norm ratio rho≥0.35；
- train 中 `Q_ratio>1 AND D_ratio>0`，且 rho ratio>0；
- H2 的 `D_ratio permutation / rho_ratio` 两端点 family 中，至少一个与上述 practical signal 对应的 `p_holm≤0.10`。

ratio 分母必须>0；`Q_ratio` 的检验使用 `D_ratio` permutation。满足全部条件才判 H2。

### H3：checkpoint / training dynamics instability

仅在 H1/H2 不成立时判断。八个 dynamics risk 端点及固定 orientation 为：`+selected_epoch`、`+last_epoch`、`+(last−selected)`、`−selected representation std`（collapse）、`−selected validation FTV loss`（FTV pressure）、`+selected前累计 grounded exposure`、`+OLS slope(val_state_loss~epoch)`（selected至last）、`−OLS slope(val_ftv_loss~epoch)`（selected至last）。Slope 窗口含 selected/last，至少3个 epoch，以显式 covariance/variance 公式计算。

H3 anchor 要求任一 oriented rho≥0.50 且 H3 八端点 family 的 `p_holm≤0.10`；support 要求另一个不同端点 oriented rho≥0.35。同时 validation all-shared 的下列九项必须**全部** practical-small：`|D_cos|<0.10`、`|D_neg|<0.10`、`|D_mbase|<0.10`、`|rho_cos|<0.35`、`|rho_mbase|<0.35`、`|D_mfail|<0.05`、`max(Q_ratio,1/Q_ratio)<1.50`、`|D_ratio|<0.25`、`|rho_ratio|<0.35`。阈值恰好相等不算 small。满足 `NOT H1 AND NOT H2 AND anchor AND distinct support AND all-nine-small` 才判 H3。

### H4：一般 stochastic seed×fold interaction

只有 core 数据质量、25-run/17–8/8-batch 合同全部通过后，H1、H2、H3 均不满足才机械判 H4。否则输出 `UNDETERMINED_DATA_QUALITY` 并令验收失败，不得贴 H4。层级固定为 H1→H2→H3→H4，保证唯一结果；Holm p≤0.10 只作小样本诊断信号，不称 confirmatory significance。H4 的正确措辞是“未达到 H1–H3 的预注册诊断证据”，不是证明 stochastic interaction 的因果机制。

## 9. Layer-wise localization

只在五个互不重叠候选组 `stage1–4 + response_projection` 上定位。对 validation 的 FAIL−PASS cosine、`M_base` 与 norm-ratio difference 分别排序，最负 cosine、最负 margin、最高 ratio 记 rank=1，tie 使用 average rank；三 rank 的算术平均由小到大定义 conflict localization ranking。Train 对每层逐项报告方向是否复现。

若至少四个候选层都越过 cosine 或 margin practical threshold，则解释为全 encoder 广泛冲突；否则报告 score 第一层及次层，不把重叠的 `encoder_overall/all_shared` 当独立定位证据。

Fold 3 仅作预声明对照：先按 fold 聚合五个 run 的 validation all-shared run medians，再报告五个 seed、fold3−其他四折 pooled difference 与五折排名。只有 fold3 的 cosine 与 `M_base` 均严格低于其余每一折，且 `fold3−others` 的 cosine≤−0.10 或 `M_base`≤−0.10，才称特殊 signature；否则明确为未见 fold3 特异。

## 10. Training history 与 checkpoint trajectory

全部 25 个 history 用于 epoch-level loss/exposure 描述；至少机械选择 3 个 PASS 与3个 FAIL：各组选 degradation 最小、预注册中位 rank、最大。选择不读取 gradient 结果。

现有每格只有 selected/fallback/last，其中 fallback 与 selected 是同 epoch、同 model state，不算独立时间点。因此 trajectory 只在六个代表 run 比较 selected 与 last；明确写明当前资产不支持 early/每 epoch gradient trajectory，也不能恢复原 batch order、dropout mask 或 SIGReg direction。

## 11. FTV coverage

报告每 fold actual train/validation pool 的 FTV比例、每个固定 batch 的 grounded count，以及 history 可恢复的逐 epoch grounded exposure。由于 pool coverage 只随 fold 变化，相关分析以 fold-level n=5 为主，25 行重复值只作描述，不用于伪造显著性。

四个固定 fold-level 端点为：fold median base degradation vs train pool FTV proportion、validation pool FTV proportion、train batch FTV-count SD、validation batch FTV-count SD。它们复用 crossed bundle 的20,000组 fold draws 做 percentile CI，并用完整 `5!=120` 个 fold-label permutations 给双侧 exact p；seed draws 在此不伪装成额外样本。常量 composition（预期固定配额会令 batch FTV-count SD=0）按 `constant_input` 如实报告，不改换端点。run-level cumulative grounded exposure 已作为 H3 的预声明 dynamics 端点，不在 coverage 表重复制造独立样本。

## 12. 固定图表与交付

至少生成以下 14 张图：

1. base degradation vs static FTV gain；
2. base degradation vs observed ΔFTV gain；
3. base degradation vs all-shared gradient cosine；
4. base degradation vs `M_base`；
5. PASS/FAIL cosine；
6. PASS/FAIL `M_base`；
7. PASS/FAIL norm ratio；
8. layer-wise cosine heatmap；
9. layer-wise `M_base` heatmap；
10. seed×fold conflict heatmap；
11. 3 PASS/3 FAIL loss trajectories；
12. conflict vs selected epoch；
13. FTV coverage/exposure vs degradation；
14. representative selected→last conflict change。

机器表固定行数与唯一键至少包括：

- `metrics/run_level_existing_metrics.csv`：25行，键 `seed_base,fold`；
- `metrics/training_history_audit.csv`：161行，键 `seed_base,fold,epoch`；
- `metrics/representative_runs.csv`：6行，键 `seed_base,fold`；
- `metrics/source_manifest.csv`：126行（20个直接冻结 upstream 文件、25 G1 selected、25 G3 selected、25 G3 history、25 G3 selection、6 representative last），键 `artifact_id`；
- `configs/audit_batch_manifest.csv`：80行；private membership 2560行且禁止提交；
- `metrics/asset_manifest.csv`：31行（25 selected +6 representative last），键 `seed_base,fold,checkpoint_kind`；
- `metrics/batch_gradient_metrics.csv`：2800行，键 `seed_base,fold,checkpoint_kind,split,batch_id,group`；
- `metrics/trajectory_batch_gradient_metrics.csv`：672行 representative last；
- `metrics/layer_level_conflict_metrics.csv`：350行，键 `seed_base,fold,split,group`；
- `metrics/run_level_conflict_metrics.csv`：50行 all-shared，键 `seed_base,fold,split`；
- `metrics/pass_fail_comparison.csv`：84行，2 splits×7 groups×6预声明指标；
- `metrics/gradient_correlation_metrics.csv`：56行，2 splits×7 groups×4核心相关；
- `metrics/phase_a_gain_correlations.csv`：3行；`metrics/dynamics_correlations.csv`：8行；
- `metrics/ftv_coverage_metrics.csv`：10行；`metrics/coverage_correlations.csv`：4行（fold-level n=5）；`metrics/layer_localization_metrics.csv`：5行；`metrics/fold_signature_metrics.csv`：10行；
- `metrics/trajectory_conflict_metrics.csv`：168行（selected/last×6×2×7）；`metrics/trajectory_change_metrics.csv`：84行；
- `metrics/hypothesis_decision.csv`：4行且恰一项 selected；`metrics/figure_manifest.csv`：14行；
- `metrics/analysis_input_manifest.csv` 绑定两张正式 gradient matrix、五张 aggregate 与既有/Phase-A 输入的 SHA、行数和列 schema；
- `metrics/final_analysis_manifest.json` 登记正式分析输出 SHA，`metrics/FINAL_ANALYSIS_COMPLETE.json` 作为最后写入的完整性 marker；
- `metrics/diagnosis.json` 与独立 `reports/acceptance.json`；同时写同内容兼容镜像 `metrics/acceptance_check.json`，两者均不得覆盖。

每张表的 exact column whitelist 与上述 Cartesian product 在冻结 config/source contract 中登记；任何缺行、额外行、重复键或公开 patient/absolute-path 字段均失败。`layer_level` 保留7组作描述，但 layer localization 只使用五个不重叠组。

最终中文报告必须逐条回答目标列出的十个问题，并把 post-hoc association、objective-level evidence 与因果限制分开。

## 13. 唯一推荐映射与停止条件

- H1：第一优先 PCGrad；第二优先 gradient normalization；
- H2：第一优先 gradient normalization；第二优先 grounding warm-up；
- H3：第一优先 checkpoint averaging；第二优先 grounding warm-up；
- H4：第一优先固定 batch order/composition 的 stochastic replicate；第二优先 checkpoint averaging。

本轮只输出推荐，不执行任何修复。最终报告和验收完成后停止。
