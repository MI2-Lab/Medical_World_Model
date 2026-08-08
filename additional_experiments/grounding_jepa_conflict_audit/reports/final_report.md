# Grounding–JEPA Conflict Audit 最终报告

> **冻结后语义勘误：** 冻结实现曾把上游 D（base degradation）误作 delta_ftv_delta_r2，产生rho=1的伪自相关。本版已从冻结的 probe_seed_fold_cell_metrics.csv 按每个seed×fold的三个change cell重算R² gain；没有重算梯度或改变H1–H4规则。详见[POST_FREEZE_SEMANTIC_ERRATUM.json](../POST_FREEZE_SEMANTIC_ERRATUM.json)。

> 本报告由冻结机器表和 `diagnosis.json` 机械生成；数值未手工转录。所有相关与检验的正式推断单位均为seed×fold run，而非batch。

## 1. 科学问题

本审计只回答：G3的base-loss instability是否与Direct FTV grounding和JEPA完整base objective在shared image representation上的梯度关系一致。它是诊断实验，不是方法改进实验。

## 2. 现有 multi-seed evidence

固定5个training seed×5 folds共25个run；state-loss safety gate得到17 PASS/8 FAIL。base degradation中位数=0.030，范围[-0.043, 0.181]。既有static FTV gain中位数=0.053，observed ΔFTV Spearman gain中位数=0.083。

关联图：[图1](../figures/01_base_degradation_vs_static_ftv_gain.png)、[图2](../figures/02_base_degradation_vs_observed_delta_ftv_gain.png)。

## 3. 为什么怀疑 gradient conflict

Grounding representation gain已在现有multi-seed结果中重复出现，但同一机制的25格仍有8格越过5% state-loss degradation阈值。因此需区分：shared-gradient方向冲突、scale imbalance、checkpoint/training dynamics，或未被简单端点解释的seed×fold stochastic interaction。

## 4. Shared computational graph

DCE7经四stage 3-D encoder、GAP和response projection得到observed response state。JEPA分支使用projector/causal transition并形成完整 `L_base = state_loss + 0.09×SIGReg`；FTV分支使用线性head形成raw patient-mean SmoothL1。真实shared参数仅为encoder（892,672）与response projection（25,152），all-shared共917,824。projector/transition、FTV head与EMA target均不进入cosine。

**安全指标与梯度目标不可混同：** 17/8 gate及base degradation使用validation `state_loss`；本审计的 `g_base` 则是训练定义的完整base objective（state loss加SIGReg）。

## 5. Audit batch protocol

每fold分别固定8个train与8个validation batch，batch size=32，每批至少8名FTV available患者；同fold五个training seed复用完全相同composition。公开manifest共80行，不含患者标识。Validation池跨batch均衡复用是预注册设计，batch从未当作独立推断样本。

每个checkpoint/batch分别执行full base与raw FTV的独立forward/backward，固定随机流，不创建optimizer且不step。controlled `model.train()`图保留transition dropout，但并不重放原训练RNG。

## 6. Overall gradient cosine

Validation固定batch中，run-level负cosine比例的均值为0.360，run median cosine的总体中位数为0.123；这量化了冲突频率，不把接近零的方向自动称为因果不兼容。

Train负cosine比例均值=0.400。degradation关联见[图3](../figures/03_base_degradation_vs_all_shared_cosine.png)，PASS/FAIL分布见[图5](../figures/05_pass_fail_cosine.png)。Train与validation是同25个模型上的composition sensitivity，不是两个独立样本。

## 7. Gradient norm ratio

Validation FAIL−PASS norm-ratio差为0.172，FAIL/PASS median ratio=1.923，degradation相关ρ=0.214（Holm p=0.313）。

分布见[图7](../figures/07_pass_fail_norm_ratio.png)。R是 `||0.25g_FTV||/||g_base||`，它描述一阶梯度scale；即使R较大，也不能单独证明高阶curvature机制。

## 8. Base-descent margin

M_base<0确实发生的比例：validation全部固定batch=0.000，train=0.000；validation PASS均值=0.000，FAIL均值=0.000。

`M_base<0`表示联合梯度的一阶方向会增加完整base objective；`0<M_base<1`表示仍下降但被削弱。关联与分布见[图4](../figures/04_base_degradation_vs_mbase.png)、[图6](../figures/06_pass_fail_mbase.png)。该完整objective证据不同于state-loss safety gate。

## 9. PASS vs FAIL

FAIL相对PASS的四个方向性端点有2/4朝更严重冲突方向：D_cos=-0.048，D_neg=0.045，D_mbase=0.008，D_mfail=0.000。

Validation cosine permutation p=0.557，M_base permutation p=0.770，norm-ratio permutation p=0.098。PASS/FAIL差值CI来自metric与gate同步抽cell的crossed seed×fold bootstrap，而不是把17/8两组独立重抽。这些p值属于预注册小样本诊断，不称confirmatory significance。

## 10. Layer-wise localization

未达到至少四层同时越阈的广泛冲突定义；定位rank前两位为 `encoder_stage_1`（score=2.333）与 `encoder_stage_2`（score=2.333）。

Layer热图见[图8](../figures/08_layerwise_cosine_heatmap.png)、[图9](../figures/09_layerwise_mbase_heatmap.png)；seed×fold all-shared对照见[图10](../figures/10_seed_fold_conflict_heatmap.png)。`encoder_overall`与`all_shared`因重叠只作描述，不作为独立定位候选。

## 11. Base degradation vs grounding benefit

既有grounding benefit与degradation的post-hoc关联为：static gain: ρ=0.058, Holm p=1.000；observed ΔFTV Spearman gain: ρ=0.200, Holm p=0.987；observed ΔFTV R² gain: ρ=0.145, Holm p=1.000。 这些是trade-off的间接关联，不是优化干预。

关联来自同一批既有25个run，不能据此推断“更强grounding导致base损伤”。

## 12. Checkpoint/training dynamics

八个预注册dynamics端点中，|oriented ρ|最大的端点为 `post_selected_ftv_improvement`：ρ=-0.379，Holm p=0.557。代表history见[图11](../figures/11_representative_loss_trajectories.png)，selected epoch与conflict的描述性关系见[图12](../figures/12_conflict_vs_selected_epoch.png)。

六个代表run从selected到last的validation cosine变化中位数为0.009，M_base变化中位数为-0.008。 配对图见[图14](../figures/14_selected_to_last_conflict_change.png)。当前资产只有六个代表run的selected→last两点；fallback与selected同state，不是独立时间点。没有early/逐epoch checkpoint，因此不支持epoch-level gradient trajectory。

原训练没有保存足以bit-exact恢复minibatch顺序、dropout mask、SIGReg direction与DataLoader RNG的状态；不得把本controlled post-hoc图解释成原训练轨迹重放。

## 13. FTV sample exposure

Train pool FTV比例跨fold范围[0.330, 0.363]；validation范围[0.430, 0.570]；固定配额使下列端点为constant_input：train_batch_ftv_count_sd、validation_batch_ftv_count_sd。见[图13](../figures/13_ftv_coverage_exposure_vs_degradation.png)。Coverage关联单位是fold（n=5），不把五个seed复制成25个独立coverage观察。

## 14. Competing hypotheses 判断

预注册层级唯一选择 **H4**。H4若被选中只表示未达到H1–H3预注册证据，不证明stochastic interaction的因果机制。

### 十个必须回答的问题

| 编号 | 机械回答 |
|---|---|
| Q1 Shared gradient是否经常冲突？ | Validation固定batch中，run-level负cosine比例的均值为0.360，run median cosine的总体中位数为0.123；这量化了冲突频率，不把接近零的方向自动称为因果不兼容。 |
| Q2 FAIL是否更严重？ | FAIL相对PASS的四个方向性端点有2/4朝更严重冲突方向：D_cos=-0.048，D_neg=0.045，D_mbase=0.008，D_mfail=0.000。 |
| Q3 degradation是否定量相关？ | Validation degradation相关为 cosine ρ=0.086、M_base ρ=0.165；对应Holm p分别为1.000与1.000。 |
| Q4 grounding越强是否损伤越重？ | 既有grounding benefit与degradation的post-hoc关联为：static gain: ρ=0.058, Holm p=1.000；observed ΔFTV Spearman gain: ρ=0.200, Holm p=0.987；observed ΔFTV R² gain: ρ=0.145, Holm p=1.000。 这些是trade-off的间接关联，不是优化干预。 |
| Q5 冲突定位在哪里？ | 未达到至少四层同时越阈的广泛冲突定义；定位rank前两位为 `encoder_stage_1`（score=2.333）与 `encoder_stage_2`（score=2.333）。 |
| Q6 FTV gradient是否过强？ | Validation FAIL−PASS norm-ratio差为0.172，FAIL/PASS median ratio=1.923，degradation相关ρ=0.214（Holm p=0.313）。 |
| Q7 M_base<0是否发生？ | M_base<0确实发生的比例：validation全部固定batch=0.000，train=0.000；validation PASS均值=0.000，FAIL均值=0.000。 |
| Q8 Fold3是否特殊？ | Fold3 validation cosine是否严格最差=False，M_base是否严格最差=False，最终special signature=False。 |
| Q9 最符合哪个hypothesis？ | 预注册层级判定唯一选择 **H4**；该标签来自机械H1→H2→H3→H4规则。 |
| Q10 下一步测试什么？ | 唯一第一优先级：**fixed batch order/composition stochastic replicate**；第二优先级：**checkpoint averaging**。本轮未执行任何修复。 |

## 15. 下一阶段推荐

唯一第一优先级：**fixed batch order/composition stochastic replicate**；第二优先级：**checkpoint averaging**。本轮未执行任何修复。

推荐严格来自所选H1–H4的冻结一对一映射。本轮没有执行PCGrad、gradient normalization、grounding warm-up、two-stage、checkpoint averaging、adaptive lambda或任何其他修复。

### 证据边界

本审计同时包含：(1) 既有grounding gain与degradation的post-hoc association；(2) 固定checkpoint上full base与raw FTV的objective-level gradient geometry；(3) selected→last有限trajectory。三者都不是随机化优化干预。因而报告定位的是与哪类机制一致，而不是证明FTV grounding对JEPA损伤的因果效应。
