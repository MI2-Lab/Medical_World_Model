# Legacy crop 与 grounding contract 复核

## 1. 复核目的与结论边界

本报告只冻结旧输入、旧 containment 口径和既有 representation 证据，作为本实验 C0 reference。它不包含新的 C1/C2 结果，也不授权训练模型。

结论是：旧实验失败的是 **current fixed-voxel crop 的 target observability**，不是 LD grounding。旧 FTV+LD pilot 只执行了 outcome-free Stage A；Stage B 从未获得授权，因而不存在 dual-grounding checkpoint、feature、prediction 或 `lambda_LD` 选择。详见旧 [最终报告](../../ftv_ld_dual_grounding_pilot/reports/final_report.md#L42-L60) 与 [阶段状态](../../ftv_ld_dual_grounding_pilot/metrics/stage_execution_status.csv)。

本实验继续遵守 [EXPERIMENT_PLAN](../EXPERIMENT_PLAN.md#L3-L18) 的顺序：先证明图像能观察 target，再讨论 representation，且本任务不自动执行 FTV+LD dual grounding。

## 2. Legacy tensor 与 DCE 通道

Legacy cache 每名受试者保存一个 NPZ，成员 `x` 的维度顺序为 `[visit, channel, Z, Y, X]`，固定 shape 为 `[4,8,32,96,96]`；visit 顺序为 T0、T1、T2、T3。旧 frozen contract 见 [stage_a.json](../../ftv_ld_dual_grounding_pilot/configs/stage_a.json#L12-L25)。

八个通道依次为：

| index | 定义 | 模型输入状态 |
|---:|---|---|
| 0 | pre-contrast | DCE7 |
| 1 | early post-contrast | DCE7 |
| 2 | late post-contrast | DCE7 |
| 3 | early − pre | DCE7 |
| 4 | late − pre | DCE7 |
| 5 | peak relative enhancement | DCE7 |
| 6 | washout relative enhancement | DCE7 |
| 7 | binary localization support | 不属于 strict-DCE7 image input |

通道命名由 clean contract 明确登记，见 [contracts.py](../../../ispy_jepa_tmi_clean/corejepa/data/contracts.py#L3-L15)。clean builder 的相对增强以 pre-contrast 为分母，完整公式见 [imaging.py](../../../ispy_jepa_tmi_clean/corejepa/data/imaging.py#L229-L259)。历史 legacy builder 与逐访视 crop origin 已缺失，因此 clean builder 只用于解释通道和候选几何策略，不能被追认成 legacy cache 的逐 voxel provenance。

另一个必须保留的输入差异是强度归一化：clean implementation 在 crop 后对每个 DCE 通道做 1%/99% clipping、median/IQR robust scaling，并限制到 `[-5,5]`，见 [imaging.py](../../../ispy_jepa_tmi_clean/corejepa/data/imaging.py#L215-L226)。因此改变 FOV、padding 或 context 可能同时改变强度分布；新实验必须单列 normalization/image-quality sensitivity，不能把 old/new difference 全部解释为几何效应。

## 3. DCE7 与 mask 隔离

Direct Grounded Response State 的 loader 一次读取 legacy NPZ 后，严格分离：

```text
image    = x[:, :7]    # [4,7,32,96,96]
roi_mask = x[:, 7:8]   # [4,1,32,96,96]
```

实现见 [data.py](../../direct_grounded_response_state/src/dgrs/data.py#L198-L247)。G1/G3 使用 GAP，模型签名拒绝传入 `roi_mask`；第一层卷积固定为 7 channels，见 [model.py](../../direct_grounded_response_state/src/dgrs/model.py#L45-L64) 与 [model.py](../../direct_grounded_response_state/src/dgrs/model.py#L217-L230)。G2/G4 才允许 mask 在 backbone spatial map 形成后进入 normalized occupancy-weighted mean；该函数按 support sum 归一化，并在 empty mask 时回退 GAP，见 [model.py](../../direct_grounded_response_state/src/dgrs/model.py#L67-L92)。

因此，“strict DCE7”只表示 mask channel 不进入模型 backbone/pooling；它不表示输入完全没有 ROI prior。`32×96×96` tensor 本身是 lesion-centered crop，定位步骤已经使用上游 support。旧报告对此限制有明确说明：[DGRS 最终报告](../../direct_grounded_response_state/reports/final_report.md#L34-L54)。

本实验 C1/C2 继续允许 full-resolution support 仅用于 localization、crop construction 与 audit，但禁止把 mask、mask volume、bbox size、crop dimensions、resize factor、FTV 或 LD 作为模型 channel/feature。

## 4. Legacy crop 几何

- 固定 crop：`(Z,Y,X)=(32,96,96)` voxel。
- crop 前无 spacing harmonization 或 resampling。
- raw DCE/support index order 为 `XYZ`，model/cache spatial order 为 `ZYX`。
- clean center 候选以 released T0 bbox center 为起点，按 source/target shape 的 normalized index coordinate 投影到后续 visit。
- 超出 source array 的范围做零填充。

旧 geometry 实现明确不 resize、resample 或基于 affine reorient，见 [geometry.py](../../ftv_ld_dual_grounding_pilot/src/ftv_ld_pilot/geometry.py#L1-L12)。旧 origin 没有写入 cache；pilot 因而在 clean start 各轴 `±2` voxel 范围内，将 full support crop 与 actual cached channel 7 做 bitwise exact reconstruction，见 [geometry.py](../../ftv_ld_dual_grounding_pilot/src/ftv_ld_pilot/geometry.py#L146-L274)。

旧 NIfTI reader 只读取 shape/pixdim，并以 shape/spacing heuristic 处理少量 slice-first layout；它不读取或验证 qform/sform，见 [nifti.py](../../../ispy_jepa_tmi_clean/corejepa/data/nifti.py#L10-L86)。所以旧 pilot 的 “DCE-mask shape/spacing/axis handling matched” 只能解释为 matched-spacing index-space geometry，不能扩大为 affine/world-space registration 已通过。本实验的新 physical audit 必须独立验证 sform/qform、orientation、obliquity 与 DCE-mask affine。

## 5. 数据与 provenance（公开路径写法）

旧 Stage A 使用：

- `${DGRS_DATA_ROOT}/I-SPY2/<subject>/manifest.json`：每个 visit 的 DCE NIfTI、FTV support NIfTI、shape、pixdim 与 released bbox；
- `${DGRS_DATA_ROOT}/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96`：legacy cache；
- `../../radiomics_next_change/data_audit/radiomics_patient_overlap.csv`：严格 overlap 与 legacy cache mapping；
- `${ISPY2_RAW_ROOT}/Multi-feature-MRI-NACT-Data.xlsx`，sheet `datawith4visits`：LD 等四访 measurement。

workbook 与 overlap 的 frozen SHA-256、375 人/1,500 patient×visit cohort 记录于 [stage_a_input_provenance.json](../../ftv_ld_dual_grounding_pilot/metrics/stage_a_input_provenance.json)。旧 `run_stage_a.py` 实际从 overlap 的 `legacy_dce8_cache` 列解析每个 NPZ，而不只依赖 cache-root placeholder，见 [run_stage_a.py](../../ftv_ld_dual_grounding_pilot/scripts/run_stage_a.py#L266-L307) 与 [run_stage_a.py](../../ftv_ld_dual_grounding_pilot/scripts/run_stage_a.py#L353-L377)。新实验不得复制本机绝对路径到公开资产。

full support 是 manifest 的 `ftv_mask_nifti > 0`。该文件由 inverse analysis mask `mask == 0` 派生，见 [preprocess_one_patient.py](../../../ispy_jepa_tmi_clean/data_processing/preprocessing/preprocess_one_patient.py#L409-L446) 与 [preprocess_one_patient.py](../../../ispy_jepa_tmi_clean/data_processing/preprocessing/preprocess_one_patient.py#L485-L507)。它是 FTV workflow inclusion-region proxy，不是 radiologist LD target 的 manual dense lesion segmentation。

## 6. 五项 legacy containment 指标

旧主比例为：

```text
containment_ratio = cached_support_voxels / full_support_voxels
```

计数与五个 flags 的唯一正式实现位于 [run_stage_a.py](../../ftv_ld_dual_grounding_pilot/scripts/run_stage_a.py#L439-L525)：

| 指标 | Legacy 精确定义 |
|---|---|
| `boundary_touch` | actual cached binary support 在 ZYX crop 六个 face 任一有非零 voxel；六面 OR 得到 `any_boundary_touch`。|
| `suspected_truncation` | `any_boundary_touch OR containment_ratio < 0.99 OR NOT diagnostic_support_available`。|
| `severe_truncation` | `containment_ratio < 0.90`；单独 boundary touch 不自动等于 severe。|
| `sufficient_containment` | support 可审计、`containment_ratio >= 0.99` 且无 boundary touch。|
| `exact_full_support_containment` | `cached_support_voxels == full_support_voxels`，即同网格、无 resize 前提下一 voxel 不丢。|

signed margin 使用 inclusive bbox：0 表示 full support 碰到 requested crop face，负数表示 support 已越界；各面 voxel margin 乘对应 spacing 得到 mm margin，见 [geometry.py](../../ftv_ld_dual_grounding_pilot/src/ftv_ld_pilot/geometry.py#L316-L425)。

必须注意：legacy `exact` 是 count equality，依赖 cached support 是同一 source support 的无插值裁剪子集。C1/C2 发生 resampling 后不能用 resampled mask voxel count 复刻该定义；本实验已改为 source physical domain 的 voxel-footprint containment，见当前 [实验计划](../EXPERIMENT_PLAN.md#L11-L18)。此外，`exact` 与 `sufficient` 并非集合上的严格包含关系：旧受控表中有 36 个 exact-but-touch，也有 9 个 sufficient-but-not-exact；新实验必须同时报告两者。

## 7. Legacy physical spacing 与 FOV

本节由旧实验保留的受控逐访视表重新做只读聚合，只公开 aggregate，不复制 subject rows。fixed crop 的 physical FOV 按 `96×spacing_X`、`96×spacing_Y`、`32×spacing_Z` 计算；旧实现见 [run_stage_a.py](../../ftv_ld_dual_grounding_pilot/scripts/run_stage_a.py#L574-L578)。

### 7.1 Spacing 分布

| Visit | X/Y spacing median [IQR], range (mm) | Z spacing median [IQR], range (mm) |
|---|---|---|
| T0 | 0.6641 [0.6250, 0.7031], 0.3125–1.3281 | 2.0000 [1.2000, 2.0500], 0.8000–2.5000 |
| T1 | 0.6641 [0.6250, 0.7031], 0.4492–1.3281 | 2.0000 [1.2000, 2.2000], 0.8000–2.5000 |
| T2 | 0.6641 [0.6250, 0.7031], 0.4688–1.3281 | 2.0000 [1.2000, 2.0500], 0.8000–2.5000 |
| T3 | 0.6641 [0.6250, 0.7031], 0.4688–1.3281 | 2.0000 [1.2000, 2.2000], 0.8000–2.5000 |

旧 overlap 中 X/Y spacing 相同，但不能据此假定其他 cohort 或未来数据总是 in-plane isotropic。

### 7.2 Fixed crop physical FOV

| Axis | Median (mm) | IQR (mm) | Min–max (mm) |
|---|---:|---:|---:|
| X | 63.754 | 60.0–67.5 | 30.0–127.5 |
| Y | 63.754 | 60.0–67.5 | 30.0–127.5 |
| Z | 64.0 | 38.4–70.4 | 25.6–80.0 |

375 人中，139 人（37.1%）四访 X/Y spacing 有变化；17 人（4.5%）Z spacing 有变化。因而 `32×96×96` 并不代表 patient/visit 一致的 physical FOV。旧公开 summary 还登记了每 visit 的 FOV median/min/max，见 [crop_containment_by_timepoint.csv](../../ftv_ld_dual_grounding_pilot/metrics/crop_containment_by_timepoint.csv)。

## 8. Legacy containment 结果

| Visit | n | Boundary touch | Suspected | Severe | Sufficient | Exact full support | Median margin (mm) | Q05 margin (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 375 | 70.7% | 70.7% | 37.1% | 29.3% | 32.0% | -6.00 | -30.00 |
| T1 | 375 | 82.7% | 84.3% | 53.6% | 15.7% | 17.1% | -15.12 | -42.51 |
| T2 | 375 | 78.1% | 84.3% | 61.6% | 15.7% | 16.5% | -15.00 | -42.90 |
| T3 | 375 | 79.2% | 86.1% | 66.1% | 13.9% | 16.3% | -15.35 | -45.76 |

所有 visit 合并：

- boundary touch 77.7%（1,165/1,500）；
- suspected truncation 81.3%（1,220/1,500）；
- severe truncation 54.6%（819/1,500）；
- sufficient containment 18.7%（280/1,500）；
- exact full-support retention 20.5%（307/1,500）；
- containment ratio median 0.861、Q05 0.156；
- whole-union approximate extent retention median 0.850、Q05 0.542。

正式数值见旧 [crop containment 报告](../../ftv_ld_dual_grounding_pilot/reports/crop_containment_report.md#L24-L37) 与 [summary CSV](../../ftv_ld_dual_grounding_pilot/metrics/crop_containment_summary.csv)。

### 8.1 Large-LD subgroup

旧 large-LD flags 在每个 visit 内分别取 q75/q90，并以 `>= threshold` 包含 ties；T0/T1 汇总是 visit-specific flags 的并集，不是 pooled LD quantile或 baseline patient-level quantile。实现见 [run_stage_a.py](../../ftv_ld_dual_grounding_pilot/scripts/run_stage_a.py#L646-L731)。

| T0/T1 subgroup | n patient×visit | Suspected | Severe | Exact | Median margin |
|---|---:|---:|---:|---:|---:|
| visit-specific top quartile | 195 | 99.0% | 73.8% | 1.0% | -21.35 mm |
| visit-specific top 10% | 77 | 100.0% | 87.0% | 0.0% | -26.00 mm |

数值见 [crop_containment_by_ld_quantile.csv](../../ftv_ld_dual_grounding_pilot/metrics/crop_containment_by_ld_quantile.csv)。T0/T1 pooled `Spearman(reported LD, minimum margin)=-0.356`；full-support largest-component approximate extent 与 LD 的 rho 为 0.599，仅作 rank sanity。LD source unit 与 zero semantics 不明确，不能据此做 mm calibration；T2/T3 的 LD zero fraction 分别为 16.5% 与 32.5%。

### 8.2 Origin 与 proxy caveat

1,488/1,500 visit（99.2%）恢复 exact origin，全部 unique；12 个 unresolved visit 均为 cached-support complete miss，保守按 retention 0、severe 处理。FTV support 几何高度碎片化：旧受控表中 1,495/1,500 visit 有多个 26-connected components，component count median 48、Q95 283；largest-component/full-support fraction median 0.848、Q05 0.186。这既可能包含真实 multifocal support，也可能包含 proxy edge/island artifacts。whole-union containment 对 observability 是保守 audit，但不能直接当作 radiologist target lesion 或 morphology surface 真值。

## 9. Legacy Stage-A gate 与本实验 gate 的区别

旧预注册五项 gate 是：T0/T1 suspected ≤10%、T0/T1 top-quartile suspected ≤20%、all-visit sufficient ≥85%、T0/T1 LD-margin rho >−0.40、exact origin recovery ≥99%，见旧 [EXPERIMENT_PLAN](../../ftv_ld_dual_grounding_pilot/EXPERIMENT_PLAN.md#L76-L88)。其中三项 containment gate 失败，决策为 `NO_GO / LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP`；exact-origin 与 rho gate 通过不能抵消 observability failure。

本实验门槛更严格且用途不同：overall sufficient 与 exact 均 ≥95%，top-quartile suspected ≤5%，top-10% ≤10%，FTV retention Q05 ≥0.95，并加入 distortion、morphology、temporal consistency、causal deployability 与 no-direct-geometry-input gate，见当前 [stage_a.json](../configs/stage_a.json#L42-L60)。不得把旧 85%/20% 阈值复制到新 decision，也不得因为 oracle-union 通过就授权 deployable candidate。

## 10. 既有 FTV grounding 证据，但不是本轮训练授权

单 seed strict-DCE7 G3−G1 的 static FTV macro ΔSpearman 为 +0.0673，但 natural-scale ΔR² 为 −0.0676；observed ΔFTV macro ΔSpearman/ΔR² 为 +0.0759/+0.0546。G3 fold 3 的 validation base loss 相对恶化 9.59%，超过旧 5% safety gate，见 [DGRS 最终报告](../../direct_grounded_response_state/reports/final_report.md#L202-L219) 与 [稳定性章节](../../direct_grounded_response_state/reports/final_report.md#L180-L184)。

五 seed G3 复核得到 static dS mean +0.0572、seed t-CI `[+0.0412,+0.0731]`；dynamic dD mean +0.0894、seed t-CI `[+0.0676,+0.1112]`。但 base safety 只有 17/25 cells 通过、8/25 失败，R3=false；无 collapse，R4=true，最终为 `PROMISING BUT UNSTABLE`，见 [G3 多种子报告](../../g3_multiseed_generalization/reports/final_report.md#L10-L25) 与 [decision.json](../../g3_multiseed_generalization/metrics/final/decision.json)。

这些结果说明 Direct static FTV grounding 是可重复的 positive mechanism，同时 optimization safety 尚未解决。它们支持“若新 input contract 通过，可先做 matched G1/G3 FTV-only sanity”，但不能跳过 Stage A，也不能授权 LD、pCR、clinical、treatment 或 transition 修改。

## 11. 对新实验的执行约束

1. C0 必须保持 legacy reference，不重写其定义。
2. C1/C2 containment 必须在 source physical domain 按完整 voxel footprint 计算；resampled mask count 不能作分母。
3. `exact_full_support_containment` 与 `sufficient_containment` 必须同时报告。
4. morphology readiness 必须区分 whole union、largest component 与 proxy artifacts，新增 surface retention 和 cut-component audit。
5. full-resolution DCE 必须从 patient manifest 读取，不能从已裁剪 cache 反推。
6. visit-adaptive、T0-anchored 与 oracle-union 必须分开标记；oracle-union 永远是 `AUDIT ONLY`。
7. crop scale、bbox size、resize factor与 mask volume只用于 audit provenance，不进入模型。
8. Stage A 没有 deployable candidate 通过全部 gate 时，禁止任何 representation training；即使通过，本任务也只允许后续 FTV-only sanity，不自动测试 LD。
