# C1B Spatial Information / Pooling Bottleneck Audit — 预注册计划

冻结时间：`2026-08-10T09:30:32Z`。本计划在任何新 spatial feature export、FTV Ridge probe 或结果查看之前写入。实验只做 frozen-checkpoint diagnostic；旧实验目录、40 个正式 checkpoint、既有 feature、prediction 与 metric 全部只读。

## 1. 问题、边界与正式矩阵

本实验区分：A) final spatial map 已含 lesion/FTV 信息，但 GAP 稀释；B) 当前 encoder 未形成足够的局部 response feature；C) 两者并存；D) padding/source geometry/resampling nuisance 是否占据 response representation。

禁止重新训练 encoder、JEPA、transition、grounding head；禁止修改 input/crop、lambda、spacing、registration 或训练预算；禁止 LD/SPH/BPE、pCR、clinical/treatment、PCGrad。FTV/support 只用于 frozen probe、oracle diagnostic 与分层，不进入 checkpoint 或模型 forward。

正式资产固定为旧实验 `c1b_overlap_eligibility_ftv_stageb` 的：

- seeds `2026,3026` × folds `0..4` × arms `L1,L3,N1,N3` = 40 个 `formal_4x8_restart1/.../selected.pt`；
- 同一 808 人 fold-assigned population、同一 train/val/test rows；139 名 train-only patient 不进入 feature/probe；
- 同一 375 人 formal FTV cohort；
- 同一 Stage-A GO、fold、FTV、observability、cache 与 target-transform contract。

上游 completion SHA-256 冻结为：matrix `0adbcf7daf74f31e70c64c1ec9a5bb259411792fb0dfa4d093ee9d3e3210b4a2`；formal feature export `f8bc1a158c93c0563b11e46cb02c4b0ef5681048febd94ab6d674d3ea4fdc40d`；postprocessing `4a599f5d76482677056f9df11e46faa1b8d4f277eedabb63d60306531e841558`；aggregation summary `2c70c429d7b32640160f8ffbbf9b3f3f7b991838227a964387bd2f4090e445c4`；data contract `dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27`。冻结脚本还必须逐一记录 40 checkpoint 和 40 formal P0 reference assets 的实时 SHA-256，并在运行前、后复核。

## 2. Encoder 与 feature-map contract

Primary map 是 online encoder 的最后一个 spatial output、GAP 的直接输入：

`DCE7 → four residual blocks → F ∈ R^(128×D×H×W) → pooling → frozen Linear(128,192)+LayerNorm → r`。

- 三个 stride-2 residual blocks 给出 effective jump/stride `8` input voxels；feature-center offset 为 input voxel-center index `0`；最大 theoretical receptive field 为 `47×47×47` input voxels。
- Legacy input `[7,32,96,96] ZYX` → final map `[128,4,12,12]`。
- C1B input `[7,112,176,160] ZYX` → final map `[128,14,22,20]`。
- C1B spacing XYZ `0.9,0.9,2.0 mm`，故 final feature jump XYZ `7.2,7.2,16.0 mm`。Legacy 沿用每 visit native spacing，Table 1 报告其 pooled median/IQR，而不伪造统一物理 spacing。

每种 single-view pooling 先生成 128-D pooled vector，再经过同一 checkpoint 的 frozen online `response_projection` 得到 192-D `r_P`。不得调用 projector、transition、target encoder/projection 或 FTV head。Primary local-global state 为 `concat(r_PLOCAL,r_P0)`，维度 384。

条件式 secondary S3 只在 final-stage oracle 未通过 Strong Oracle Recovery 时执行。S3 固定为 `encoder.features[2]` 的 output：channels `64`、jump `4`、offset `0`、theoretical RF `23³`；不检查其他 stage。

## 3. 预注册 pooling

### P0 — existing GAP

`r_P0 = response_projection(mean(F, D/H/W))`。它必须逐元素复现 immutable formal `online_preprojector_r`。

正式 STOP gate 覆盖 40/40 cells × 808 patients × 4 visits：identity/order/split/checkpoint SHA 完全一致；shape `[808,4,192]`；finite；`torch/numpy allclose(rtol=1e-5, atol=1e-6)` 通过率 100%。记录 max/mean absolute error、RMSE、bitwise-equal fraction。任一 cell 失败，停止所有 FTV/nuisance probes。

P0 probe 还必须复现旧 formal Ridge：selected alpha、target transform、row counts 与 patient/test keys exact；natural/analysis prediction 使用 `rtol=1e-5, atol=1e-6`；pooled natural metrics absolute difference ≤`1e-6`。失败则停止解释 alternate pooling。

### PVALID — valid-source receptive-field pooling

只对有 source-authoritative valid-source mask 的 C1B/N arms 定义。对 input mask `M∈[0,1]`：

`w_valid = avg_pool3d(M, kernel=47, stride=8, padding=23, count_include_pad=True)`。

这给每个 final feature location 的固定 `47³` theoretical receptive-field 中真实 source fraction；crop 外 neural padding 与 crop 内 outside-source 均在固定分母中按 0。`r_PVALID = response_projection(sum(wF)/sum(w))`。权重、分母或几何量绝不拼接给 probe。

Legacy cache 没有 source-authoritative valid mask，historical crop origin 也非全量可恢复。因此 L1/L3 的 PVALID rows 固定为 `NA_no_source_authoritative_mask`；禁止用 lesion channel、`image==0` 或全一 mask 替代。Table 2/3 保留矩形 NA rows，但不制造数值。

### PLOCAL — fixed central physical 64-mm pooling

固定窗口为 C1B frozen coordinate system 中、以 crop physical center 为中心的 `64×64×64 mm` cube；不读 lesion/FTV，不按 visit recenter，不调窗口。

feature location `j` 的中心由 exact jump/offset 映射到 input physical coordinate；其 sampling cell 固定为每轴 `8` 个 input voxels宽。`w_local` 是该 physical sampling cell 与 central cube 的三轴 fractional volume overlap；`r_PLOCAL = response_projection(sum(wF)/sum(w))`。该定义不把 64-mm window 扩散成 47³ RF，保持 local view 语义。

C1B 使用 frozen grid affine；Legacy 使用同 visit source affine 的 voxel spacing与 crop中心相对坐标，不声称恢复缺失的 absolute crop origin。窗口始终 outcome-free、center-fixed。

### PLOCAL+GLOBAL — primary local-global state

`r_PLG = concat(r_PLOCAL,r_P0)`，无 fusion network、无训练，直接进入同一 Ridge protocol。预注册 secondary sensitivity `concat(r_PLOCAL,r_PVALID)` 仅限 N arms，标记 `PLOCAL+PVALID_SECONDARY`，不替代 primary decision。

### PORACLE — diagnostic lesion-support pooling

仅在 formal FTV/support subset 使用。C1B source support 从 frozen manifest 路径读取，经原 hash-locked RAS+ loader 与 C1B-H identity transform重采样至 frozen input grid；source canonical hash、NN target positive count、source/retained count与volume必须逐 visit复现 cache sidecar。

Legacy 1500 个 formal visits 中只有1488个可由 immutable cache第8通道定位，另12个是 `complete_miss`、cached support为空且历史crop origin不可权威恢复。为保持完整且一致的OOF人口，L1/L3 PORACLE整体固定为 `NA_incomplete_source_authoritative_support`；不得只删12行、以P0 fallback或从FTV数值伪造位置。1488/1500 availability只作描述性provenance。Table 2/3保留矩形NA rows。

N1/N3 的 `w_lesion` 使用与 PVALID 相同的 47³ exact theoretical-RF occupancy；`r_PORACLE = response_projection(sum(wF)/sum(w))`。这两个C1B arms的目标行必须全部有非空 support；不得 fallback P0、填 NaN 后静默删人口或改变 OOF denominator。PORACLE 只作 upper-bound diagnostic，禁止作为 deployment 方法或训练输入。

Mask→feature validation 在 probe 前完成：全量 shape/range/nonempty/count/hash checks；另用 outcome-free deterministic hash selection保存若干 input mask→feature occupancy overlay。montage 不参与 pooling/window选择。

## 4. Probe contract

Static 与 literal ΔFTV 逐 pooling 独立运行，但严格复用旧 `ALPHAS`、`select_ridge`、`fit_static_probe_transform`、`static_targets` 与 `literal_delta_targets`：

- feature StandardScaler 只 fit outer train；alpha grid `[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]`；validation analysis-space MSE选 alpha，`best+1e-12` 内取最小 alpha；不以 test refit或selection；test predict恰一次。
- Static transform在 outer-primary-train observable visits pooled fit：`log(FTV+epsilon)`、1/99 winsor、median/IQR；Ridge fit transformed target，再 inverse到 natural FTV。
- Δfeature=`r[t+1]-r[t]`；ΔFTV=`FTV[t+1]-FTV[t]`；禁止 ΔFTV supervision/log/ratio。
- Primary scope为 measurement-valid；observable-only 为 sensitivity。
- 报 T0–T3/三 transition 与 macro。Natural OOF先汇合五个 outer-test folds再算 metric；macro为 endpoint metrics非加权均值。不同 fold transformed scale不得伪装成 pooled transformed metric，只报 outer-fold rows或明确的 fold-metric mean。
- Metrics：Spearman、Pearson、natural R²、transformed/standardized R²、RMSE、MAE、B0 gain、prediction/target variance ratio、`Cov(y_true,y_pred)/Var(y_true)` descriptive calibration slope。

## 5. Recovery 与 prospective gates

对 seed `s`，N1 primary：`D_s = rho(L1,P0)-rho(N1,P0)`；`Recovery_s(P)=[rho(N1,P)-rho(N1,P0)]/D_s`。若 `D_s≤0`，ratio记 NA、只报 absolute effect。N3以 matched L3作重复性分析，标 secondary。

Strong Oracle Recovery（pilot）：N1在两个 seeds 均 `PORACLE-P0 static macro Spearman ≥0.15` 且 `Recovery(PORACLE)≥0.50`；N3用matched L3 deficit作复核，但不改变gate。

Deployable Local Recovery（pilot）：两个 seeds 中同一个 `PLOCAL` 或 `PLOCAL+GLOBAL` 均满足 `static macro Spearman gain ≥0.10` 或 `Recovery≥0.33`。

Padding/geometry evidence gate：两个 seeds 的 N1 PVALID 均满足 gain `≥0.10` 或 recovery `≥0.33`；同时 P0 对 `padding_fraction` 或 `valid_source_fraction` 的 macro OOF `R²≥0.20`，且 PVALID 相对P0在同 target 的 R²降低 `≥0.10`，两个 seeds均成立。阈值只作 prospective classification；完整方向和数值必须全部报告，结论仅写 `SUPPORTED IN PILOT`。

## 6. Nuisance、occupancy、downsampling 与 budget

Outcome-free nuisance targets固定为：padding_fraction、valid_source_fraction、native_spacing_x/y/z、acquisition_fov_x/y/z、max_resample_factor、resize_anisotropy。后者定义为三个 `source_samples_per_output_axis` 中 `max/min`；spacing来自 source affine列范数；FOV=`shape×spacing`。对 P0/PVALID/PLOCAL 使用相同 outer-fold Ridge和单次test规则，每 visit endpoint独立，报告 endpoint与macro Spearman/R²。Legacy PVALID仍为NA。

Lesion occupancy固定为：cache中 `support_retained_source_volume_mm3 / (valid_source_voxels × abs(det(C1B grid affine linear)))`。仅1500 formal support visits；pooled distribution用 `qcut(...,4)`得到Q1–Q4，若边界非唯一则STOP而不改规则。quartile只 join 已冻结 OOF P0 predictions，不重新fit。逐seed/endpoint/quartile报告 L1/N1 Spearman、MAE和差；另报告 `Spearman(occupancy, |e_N1|-|e_L1|)`。

Downsampling只 join frozen P0 OOF errors，bins固定 `≤1.5`, `(1.5,2]`, `>2`，报告count、error difference与相关性；不据此修改spacing。

Training budget只读 history/selection。逐cell报告 selected epoch、observed max epoch、是否hit configured epoch 12、selected是否位于最后2个observed epochs、final-selected state loss、最后3个 finite validation state-loss的OLS slope、stop reason与trajectory。某N arm仅在以下全部满足时标 `UNDERTRAINING_PLAUSIBLE`：≥60% cells hit epoch 12；≥60% cells selected in last two；median normalized last-three slope `≤-0.005` per epoch。它只能作secondary confound，不能覆盖A–D。

## 7. Conditional S3 与唯一分类

若 final-stage Strong Oracle Recovery不成立，必须执行唯一允许的 S3 secondary audit，重复 P0/PLOCAL/PORACLE、同一 probe和 recovery thresholds；否则明确记录 `NOT_TRIGGERED_FINAL_ORACLE_STRONG`。

唯一 A–D 选择按以下顺序：

1. final oracle弱且S3 oracle也未过Strong gate → **C ENCODER BOTTLENECK**；NEXT=`Stronger Pretrained 3-D Encoder Pilot`。
2. Padding/geometry evidence gate成立 → **B PADDING / GEOMETRY DILUTION**；NEXT=`Valid-source-aware + Localized Response Pooling Pilot`。
3. Strong Oracle + Deployable Local成立 → **A POOLING BOTTLENECK**；NEXT=`Local–Global Response State Pilot`。
4. 其余（含oracle强但mask-free local弱，或S3强/final弱，或多机制部分成立）→ **D MIXED BOTTLENECK**。NEXT依证据唯一映射：oracle强而所有mask-free弱=`Learned Spatial Response Aggregation Pilot`；S3强/final弱=`Preserve Higher-Resolution Spatial Features Pilot`；否则=`Local–Global Response State Minimal Pilot`。

无论结果如何，不回退legacy fixed voxel crop，不把PORACLE作为方法，不在本audit训练下一方案。

## 8. Deliverables 与隐私

Public：Table 1 feature-map contract；Table 2 static FTV；Table 3 observed ΔFTV；Table 4 legacy deficit recovery；Table 5 nuisance decodability；Table 6 occupancy stratification；Table 7 training budget。至少12图：schematic、pooling illustration、static rho、static R²、delta rho、recovery、oracle-vs-local、padding decode、FTV-vs-nuisance、occupancy degradation、budget、activation montage。

中文 `reports/final_report.md` 必须逐条回答用户14问、给唯一A–D和唯一NEXT。所有 patient IDs、paths、OOF rows、feature states、support/valid masks、checkpoint与private manifests保持owner-only并由 `.gitignore` 排除；公开资产只含aggregate和hash。最终运行public privacy scan、相对链接检查、old-tree immutability hash复核、40/40 inventory与test suite。
