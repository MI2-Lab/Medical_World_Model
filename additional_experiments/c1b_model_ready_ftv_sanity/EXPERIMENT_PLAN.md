# C1B Model-Ready + FTV-only Response-State Sanity：冻结计划

## 1. 科学问题与禁止项

本实验只检验：修复 legacy partial-lesion observability 后，严格 DCE7 observed state 对 static FTV、literal observed `Delta FTV = FTV_(t+1) - FTV_t` 的表达，以及 JEPA/base optimization safety 是否改善。

禁止 LD、SPH、BPE supervision，禁止修改 transition，禁止 clinical/treatment/pCR supervision，禁止用未来 lesion/support 修补当前输入，禁止让 mask、bbox、crop scale、spacing、affine、valid-source mask 或任何几何 metadata 进入模型 tensor。pCR 不运行。旧实验和旧 cache 均只读。

## 2. 两级 hard gate

- Stage A：完成 raw-DICOM pixel rebuild、真实 RAS+ canonicalization、C1B-H/C1B-R image-only registration sensitivity、downsampling audit、完整 3-D DCE7 builder 与 cache round-trip。
- 只有 `STAGE_A=GO` 才可启动 Stage B 的 L1/L3/N1/N3 训练。任何 Stage A 硬门失败均写成 `NO-GO` 并停止；不得降低阈值。

## 3. Cohort 与 split

- 正式 observability/FTV QC cohort：既有 375 人、T0-T3 共 1,500 visit。
- 为与 Direct Grounding base objective 公平配对，正式训练 population 保留 locked seed-2026 manifest 的 808 名 complete-four-visit I-SPY2。此前仅参与 base objective 的 156 名 I-SPY1 是预冻结 source-QC 候选；严格、outcome-free raw-DICOM 审计后仅 140 名四访均 PASS，因此 model-ready/Stage B population 固定为 `808+140=948`，其余16名 fail closed。该裁决只读 imaging source/geometry/pixels，发生于任何训练或 endpoint 之前，不按 performance 补回。
- 375 人之外的 model-input visit 也必须通过 finite affine、RAS+、DCE7 与 cache QC。若 I-SPY2 中发现正式 72 个之外的同类 singular geometry，必须一并 pixel rebuild，不能把坏 geometry 输入 base objective。
- I-SPY2 T0 released localization support只用于冻结 T0 physical center。无 released support 的 base-only I-SPY1 使用 outcome-free T0 acquisition physical center fallback；四访仍共用同一 T0 grid。fallback 类型只写 sidecar，不进入模型。
- split 固定为 seed-2026 five-fold candidate copy；不重划 fold。训练 seed 只用 2026、3026；effective run seed 为 `seed_base + fold`。

## 4. C1B input contract

- 唯一候选：T0-anchored fixed physical detail crop。
- canonical orientation：RAS+；必须根据 affine permutation/flip 重排数组。
- output shape ZYX：`112 x 176 x 160`。
- spacing XYZ：`0.9 x 0.9 x 2.0 mm`。
- voxel-footprint FOV XYZ：`144 x 158.4 x 224 mm`。
- T0-T3 共用同一 center、basis、shape 与 spacing。固定 grid 不按 T1-T3 bbox recenter，也不为 outlier 改 spacing/FOV；overflow 只记录为 containment failure。这一策略防止少数 outlier 改写全 cohort input scale。
- intensity 使用线性 physical interpolation；mask/support 只作 nearest-neighbor sidecar QC。任一轴 downsampling factor >1.5 时先施加按 source spacing 计算的 Gaussian anti-alias；`>2` 的 visit 逐例审计。
- source 外采样不得使用可被精确识别的 sentinel。正式 padding 候选固定为 current-volume reflection；同时保存真实 `valid_source_mask` sidecar。用 outcome-free intensity/background distinguishability audit 检查它，不把 mask 输入模型。

## 5. Raw-DICOM pixel rebuild contract

正式 72/1,500 singular-sform visit 必须全部从 PixelData 重建；为了完整 base cohort，额外发现的同类 I-SPY2 visit也执行同一流程。

每个 series 必须：唯一 SOP/Series；Rows/Columns、IOP、PixelSpacing 一致；TPI 与 AcquisitionTime 各自形成完整 `(time,slice)` 网格且一一对应；phase 顺序由 chronologically unwrapped AcquisitionTime 冻结，并要求 TPI 数值顺序同向；slice 按 `dot(IPP, cross(IOP_row, IOP_col))` 排序；每 cell 恰好一次；应用逐文件 RescaleSlope/Intercept；重建 `[X,Y,Z,T]` float32；用 IOP/IPP/spacing 生成 LPS affine 后转 RAS；逐 cell 回读并与 volume slice exact compare；finite、nonconstant；与 mask physical center和footprint corner Hausdorff 均 `<=0.1 mm`。输出 NIfTI qform/sform 均有效，保存 hash 与 private cell audit。

## 6. DCE7 phase 与 normalization contract

冻结 legacy-compatible、outcome-free semantics：

1. `pre`
2. `early`
3. `late`
4. `early - pre`
5. `late - pre`
6. `(peak - pre) / max(abs(pre), 1)`
7. `(late - peak) / max(abs(pre), 1)`

`peak=max(phases[1:])`。对 `T<=4`，early=`pre+1` clipped、late=最后一相；否则只读取既有 acquisition phase metadata 的 pre/post_early/post_late index，缺失时固定 default `0/min(2,T-1)/min(5,T-1)`。不得按 FTV、pCR、LD 或图像病灶表现选相。每个 visit 将 phase count、indices、channel names 写 sidecar。

crop 后每 channel 独立：valid-source P1/P99 clip，median 与 `(IQR/1.349)` scale，退化时用 std，最终 clip `[-5,5]`。不使用跨患者/test statistics。padding 在相同变换后保持非 sentinel；valid-source mask只用于统计/QC。

## 7. C1B-H vs C1B-R registration sensitivity

C1B-R 只使用 selected precontrast MRI，whole acquisition/stable anatomy image mask和 Mattes mutual information rigid registration；不读 lesion mask、FTV、radiomics、clinical、treatment、pCR。T1-T3 register to T0，transform只作为 resampling operator和 private QC sidecar。

若总体R策略通过但个别pair optimizer fail，预冻结的deployable disposition为该pair严格回退identity/header operator（C1B-H），不得用lesion或后验结果补transform；failure/fallback仍计入registration failure rate。若总体success低于95%，整个策略选H而不是靠fallback掩盖失败。

只有同时满足以下预冻结条件才选 R，否则选 H：

1. transform finite且成功率 `>=95%`；catastrophic transform（translation norm >75 mm 或任一 rotation >20 degrees）比例 `<=1%`；
2. whole-anatomy similarity median gain `>0.02`，至少 75% moving visits不恶化超过 0.01；
3. C1B-R available-support exact containment不比 H 低超过 0.5 percentage point，FTV retention Q05 `>=0.95`；
4. median padding不增加超过2 points、Q95不增加超过5 points；
5. 没有 `anatomy residual >5 mm` 但 lesion residual被压到 `<2 mm` 的系统性 lesion-align pattern；
6. small/medium/large lesion与最高 transform病例的 blinded技术图审无明显 ghosting、左右翻转或 deformation。

注册 contrast 不可比、图审失败或上述任一项失败即冻结 C1B-H。不得根据 Stage B performance选 registration。

## 8. Source-edge grounding observability

`grounding_observable_mask(i,t)=0` 当 FTV inclusion support触及 source acquisition任一 voxel face，或该 visit 有预注册的 source/pixel geometry不可观察状态；否则为1。该 mask 只与 measurement validity 相交生成 grounding-loss eligibility，不用于 forward、不删患者、不改变 base loss、不筛选 primary readout。

FTV grounding transform在每个 outer fold只用 train 中 `measurement_valid & grounding_observable` visit拟合。representation probes保留全部 measurement-valid visit为 primary，并另报 observable-only sensitivity。

## 9. Stage A hard gate

必须全部满足：

1. 正式72个异常visit pixel rebuild与逐cell order全通过；
2. 所有实际送入模型的 visit 有finite、可逆 affine，并真正 canonical RAS+；
3. registration策略按第7节冻结；
4. extreme resampling逐例有 disposition，且无 unresolved catastrophic case；
5. 完整3-D DCE7 shape/phase/channel/normalization验证；
6. repair病例、source-edge、large-lesion与每fold抽样 cache round-trip/hash通过；
7. 正式375人 C1B available-support exact containment `>=0.95`；
8. FTV retention Q05 `>=0.95`；
9. 无 future leakage；model tensor严格只有DCE7；
10. grounding observability定义、private manifest与公开aggregate冻结。

## 10. Stage B 2x2（仅 Stage A GO 后）

| model | input | objective |
|---|---|---|
| L1 | legacy DCE7 | existing JEPA/base |
| L3 | legacy DCE7 | same base + Direct FTV, lambda=0.25, observable mask |
| N1 | frozen C1B DCE7 | existing JEPA/base |
| N3 | frozen C1B DCE7 | same base + Direct FTV, lambda=0.25, observable mask |

模型固定为 strict DCE7 -> 3-D encoder -> GAP -> 192-D online pre-projector `r`；既有 projector与causal transition完全不变。FTV head只在L3/N3训练时读取 `r`，不参与feature extraction。禁止 lambda sweep。

四臂统一 effective batch 32、optimizer steps、patients/epoch、LR、epoch/patience、EMA、SIGReg和无augmentation规则。实测 C1B physical BS=4 可在冻结3-D encoder上通过显存smoke（约37.3 GiB），因此四臂统一 physical BS=4/accum=8；若正式run发生OOM，只允许全四臂共同降到BS=2/accum=16并重新开始，不可按input/model单独改变。effective-batch sampler先按32截断，保持共同patient order；SIGReg固定在相同physical microbatch语义，FTV按整个logical batch的patient numerator/count精确聚合，gradient clipping/optimizer/EMA每8个microbatch仅执行一次。每个 `(seed,fold)` 四臂公共模块初始化必须 hash相同。

## 11. Frozen endpoints

- Static：train-only StandardScaler + Ridge，`r_t -> FTV_t`，T0-T3及macro；Spearman、natural-scale R2/RMSE/MAE、train-mean B0 RMSE gain、prediction variance ratio。
- Observed change：`r_(t+1)-r_t -> literal FTV_(t+1)-FTV_t`，三个相邻transition及macro；同类指标。旧 `logFTV_end-logFTV_start` 只可明确标为 secondary legacy-log-difference sensitivity，不得称 natural DeltaFTV。
- Optimization：selected validation next-state loss（历史 `val_state_loss`）定义 base degradation；记录 base/FTV/total loss、epoch、representation std、finite、`>5%` fail。
- Primary comparisons：N3-N1、L3-L1、N1-L1，及 DiD `(N3-N1)-(L3-L1)`，逐seed/fold paired。

最终分类只允许 `OBSERVABILITY-SUPPORTED`、`OBSERVABILITY + OPTIMIZATION BOTTLENECK`、`INPUT FIX NOT REPRESENTATION-LIMITING`；Stage A失败则报告 `STAGE_A NO-GO / STAGE_B NOT RUN`，不伪造representation结论。
