# BPE Source Observability & Context-FOV Audit 最终报告

## 执行结论

唯一合法 scientific classification 是 **D — `BPE_SOURCE_NOT_RELIABLY_AUDITABLE`**。
原因不是 selected-series、reconstructed-image 或 C1B-cache availability 缺失：375 人 ×
4 visits 的 1,500 个 selected raw DICOM series、1,500 个 reconstructed DCE 和 375 个
frozen C1B caches 均可用。
决定性缺口是生成 BPE 标量时的真实 contralateral breast/FGT source ROI、五层坐标和
hash-bound lesion→contralateral mapping 未发布。

因此 F0 LOCAL、F1 full C1B-H、F2 reconstructed source-image 三个 observability gates 都是
**`NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE`**。这不是三次 FAIL，也不证明 LOCAL/C1B
disjoint、acquisition limited 或 representation failure。下一阶段选择是
**`PAUSE_BPE`**；本实验 **不授权** Local–Context Phenotype Representation Pilot。

## Outcome-blind firewall 与 audit population

本运行没有读取 pCR、HR、HER2、treatment、clinical model performance；BPE 数值只在
读取 workbook 后立即归约为 finite/missing availability flag，没有被保留、排序或用于
case/FOV/gate selection。20 人 geometry sample 按 salted patient-ID hash 固定抽取，
selection SHA-256 为 `24d43ae42bab9c6151e7ef918b48b1e0c6c63978ce18b64aa0815141a4908ed6`。

| Stage | Patients | Visits | Included | Reason |
|---|---|---|---|---|
| SOURCE_WORKBOOK | 384 | 1536 | True | source inventory |
| BPE_COMPLETE_WORKBOOK | 384 | 1536 | True | finite source scalar availability only |
| WORKBOOK_ONLY_EXCLUDED | 9 | 0 | False | NO_COMPLETE_FOUR_VISIT_C1B_MATCH |
| PRIMARY_MATCHED_COHORT | 375 | 1500 | True | BPE-complete exact MRI/C1B match |
| RECONSTRUCTED_DCE_AVAILABLE | 375 | 1500 | True | technical source availability |
| C1B_CACHE_AVAILABLE | 375 | 1500 | True | technical cache availability |
| RAW_DICOM_SELECTED_SERIES_AVAILABLE | 375 | 1500 | True | selected raw series availability; extent geometry audited on reconstructed source image |
| AUTHORITATIVE_BPE_SOURCE_ROI_AVAILABLE | 0 | 0 | False | SOURCE_ROI_NOT_AVAILABLE |

## 三个 input contracts 与 geometry validation

- **F0**：固定 `64 × 64 × 64 mm` LOCAL readout support。它不是 input crop；encoder
  仍读取完整 F1 tensor，不能把 receptive field 或 padding 当作 source observability。
- **F1**：固定 ZYX `[112, 176, 160]`、XYZ spacing `[0.9, 0.9, 2.0]` mm、
  XYZ footprint `[144.0, 158.4, 224.0]` mm、true RAS+、T0 anchor、不重心化。375/375
  affines/grid centers 验证通过，max center/extent error 都为
  `2.842e-14` mm。
- **F2**：reconstructed source affine 对 half-voxel bounds 定义的 source-image
  parallelepiped，并关联到 frozen selected raw series。Raw series existence 与 first-instance
  geometry tags 已核对，但没有从 full raw series 独立重建第二个 footprint；因此不声称
  raw↔reconstructed footprint equivalence。公开 extent 是各轴独立的 RAS AABB marginal
  summaries，不能用于代替 exact F2 containment，也不是一个 realizable median tensor。

Primary 1,500 visits 的 canonical orientation 前为
`{'LAS': 396, 'LPS': 72, 'RPS': 1032}`，后为 1,500/1,500 RAS+；max voxel-center corner
roundtrip error `8.039e-14` mm，DCE/mask
footprint error max `0.001836` mm。

## Primary observability results

| FOV | Eligible visits | ROI available | Coverage evaluable | Gate | Status |
|---|---|---|---|---|---|
| F0 | 1500 | 0 | 0 | NOT_EVALUABLE | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |
| F1 | 1500 | 0 | 0 | NOT_EVALUABLE | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |
| F2 | 1500 | 0 | 0 | NOT_EVALUABLE | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |

Source occupancy、X/Y/Z boundary touch、centroid inclusion、physical margin、ROI 内
valid-source/padding/extrapolation 比例和 contralateral breast coverage 都需要真实 ROI。
现有 whole-grid C1B/source overlap（1,500/1,500 >0）只证明 reconstructed source image
与 C1B 有某些交集，不能替代 BPE ROI overlap。

## F2 reconstructed source-image support 描述（不是 BPE context FOV）

| Statistic | X AABB mm | Y AABB mm | Z AABB mm |
|---|---:|---:|---:|
| Q50 | 170.7 | 170.7 | 160.0 |
| Q90 | 320.0 | 320.0 | 176.0 |
| Q95 | 340.0 | 340.0 | 193.6 |
| Q99 | 380.0 | 380.0 | 208.0 |

X/Y/Z 的每个分位数都是单独计算的 marginal statistic；同一行不代表某一真实 visit 或
一个可实现的 tensor geometry。53/1,500 source affines 是 oblique，因此 AABB 也不是 exact
parallelepiped。这些数只描述 reconstructed source image；没有 breast mask/FGT ROI 时，
不能证明 BPE source 完整、不能定义 bilateral crop，也不能被称为 CXT2 所需 FOV。

## Laterality / alignment audit

20 人 × 4 visits 的 outcome-blind sample 全部通过 RAS/affine consistency。全 1,500
visits 中 DICOM `Laterality` 有值 `1272`，
缺失 `228`；`ImageLaterality` 有值
`0`；
`1` 人的已知
`Laterality` 跨 visits 冲突。IOP/IPP/PixelSpacing/Rows/Columns/FrameOfReferenceUID 在
`1500`/1,500 first-instance headers
齐全。

但 `Laterality` 不完整，且没有 hash-bound 到 lesion side 或 historical BPE source-side
selection；RAS X 正负也不能独自定义 patient midline。结论只能是
**geometry orientation passed，BPE contralateral mapping unresolved**。

## Longitudinal geometry

| Interval | Eligible | Evaluable | ROI anchor | Status |
|---|---|---|---|---|
| T0->T1 | 375 | 0 | NOT_DOCUMENTED | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |
| T1->T2 | 375 | 0 | NOT_DOCUMENTED | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |
| T2->T3 | 375 | 0 | NOT_DOCUMENTED | NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE |

T0/T1/T2/T3 的 BPE scalar 是 visit-specific measurement；原方法没有说明真实 FGT mask
是每访视重分割、baseline anchored 还是注册传播。故 centroid displacement、physical
volume variation 与 laterality consistency 都不能用 acquisition center 或 lesion mask
代算。本实验也没有为了 temporal consistency 注册 future visit。

## Compute / resolution audit

| Contract | Role | X mm | Y mm | Z mm | Voxels/visit | MiB/visit | vs C1B | Status |
|---|---|---|---|---|---|---|---|---|
| F0_LOCAL_READOUT_SUPPORT | readout_support_not_full_tensor | 64 | 64 | 64 | 3153920 | 84.219 | 1 | READOUT_ONLY_ACTUAL_INPUT_IS_F1;hypothetical_support_voxels=165888 |
| F1_FULL_C1B_H | existing_model_tensor_support | 144 | 158.4 | 224 | 3153920 | 84.219 | 1 | EXISTING_FROZEN_CONTRACT |
| F2_RECONSTRUCTED_SOURCE_SUPPORT_MARGINAL_Q50 | separate_marginal_summaries_not_one_realizable_tensor | 170.667 | 170.667 | 160 | 5242880 | 20 | 1.662 | RECONSTRUCTED_SOURCE_SUPPORT_ONLY;RAW_FOOTPRINT_EQUIVALENCE_NOT_RECOMPUTED;NOT_BPE_CONTEXT_CANDIDATE |
| CXT1_FULL_C1B_CONTEXT | candidate_only_if_F1_passes | NA | NA | NA | NA | NA | NA | NOT_AUTHORIZED_F1_NOT_EVALUABLE |
| CXT2_BILATERAL_BREAST_PHYSICAL_CONTEXT | candidate_only_if_F1_fails_and_F2_passes | NA | NA | NA | NA | NA | NA | NOT_DEFINED_SOURCE_ROI_AND_BREAST_SUPPORT_NOT_AVAILABLE |

F1 为 `3,153,920` voxels/visit；DCE7 float32 原始 image tensor 为
`84.219 MiB/visit`，四访视为
`336.875 MiB/patient`，未计 gradients、activations 或 framework overhead。
F0 只是 readout，因此实际 input cost 仍为 F1 的 1.0×。64-mm support 若仅按 C1B spacing
离散化会是 `32×72×72=165,888` voxels，但这不是当前 model input。

F2 的 5,242,880 voxels/visit 与 20 MiB 是 source voxel count 各自 Q50 所导出的描述性
single-float32-volume lower bound；它与 X/Y/Z marginal Q50 不共同定义一个真实 visit，
也不是 DCE7 context tensor cost。由于 CXT2 没有被可靠定义，不能给 production tensor、
downsampling factor 或 `MULTISCALE_CONTEXT_RECOMMENDED`。

## 十四个问题逐项回答

### 1. BPE 真实来源是什么？

Background parenchymal enhancement：对侧乳腺在 S–I 方向几何居中的连续五个 axial
slices 中，自动 breast segmentation + fuzzy c-means 得到的 fibroglandular tissue 的
mean early percent enhancement。不是 ipsilateral lesion ROI、late PE、SER 或 manual ROI。

### 2. Source ROI 是否可恢复？

否。状态为 **`SOURCE_ROI_NOT_AVAILABLE`**。Scalar semantics 可恢复，historical voxel
source 不可恢复；不得伪造 proxy ROI。

### 3. LOCAL 覆盖多少？

**NA / NOT EVALUABLE**。64-mm support 已冻结，但没有 source ROI 可与其做 physical
intersection。

### 4. Full C1B 覆盖多少？

**NA / NOT EVALUABLE**。F1 geometry 完整且 1,500/1,500 visits 有某些 valid-source
overlap，但这不是 BPE ROI occupancy。

### 5. 原 acquisition 覆盖多少？

**NA / NOT EVALUABLE**。Reconstructed F2 source-image support 可恢复且 selected raw series
存在，但本 audit 未独立重建 raw full-series footprint equivalence；更关键的是 BPE source
completeness 不可恢复，image boundary 不能冒充 breast boundary。

### 6. 是否有 boundary-touch？

未知。F0/F1/F2 的 BPE source boundary-touch 均不可计算。

### 7. 是否有左右侧映射风险？

有。True RAS conversion 已通过，但 patient-specific lesion→contralateral source mapping
不存在；DICOM laterality tag 又不完整且非 authoritative BPE provenance。

### 8. Longitudinal geometry 是否稳定？

不可判断。三个 interval 的 centroid/volume/laterality rows 均 0 evaluable patients。

### 9. Full C1B 是否已经足够？

不可判断，不能归类为 A `FULL_C1B_CONTEXT_SUFFICIENT`。

### 10. 是否需要 bilateral broader context？

不可判断，不能归类为 B `BROADER_BILATERAL_CONTEXT_REQUIRED`；nominal anatomy 语义
不能替代 ROI coverage gate。

### 11. Broader context 大约需要多大物理 FOV？

当前不能定义。上表 F2 Q50/Q90/Q95/Q99 仅为 reconstructed source-image AABB 的独立
marginal statistics，不是所需 bilateral breast FOV、单一 tensor geometry 或 crop proposal。

### 12. 计算代价如何？

F0 实际仍使用 F1 input；F1 为 84.219 MiB/visit、336.875 MiB/four visits 的 DCE7
float32 image tensor。CXT2 未定义，故没有诚实的相对 production cost。

### 13. 下一阶段 phenotype branch 应该使用什么？

选择 **暂停 BPE（`PAUSE_BPE`）**，不是 current LOCAL、full C1B 或 bilateral context。
先恢复 authoritative source ROI、five-slice mapping 与 lesion-side provenance，再原样重跑
本实验 gates。

### 14. 本实验是否授权下一阶段 context representation pilot？

**否。** 只有分类 A 或 B 才授权 Local–Context Phenotype Representation Pilot；本实验
是 D。也不能跳到 phenotype-specific representation audit，因为那要求先证明 C：LOCAL
already observable。

## Scientific classification 与授权

- Classification：**D — `BPE_SOURCE_NOT_RELIABLY_AUDITABLE`**
- Next input：**`PAUSE_BPE`**
- Context candidate：`null`
- Local–Context Phenotype Representation Pilot：**NOT AUTHORIZED**
- Machine-readable decision：`metrics/decision.json`

## Git / delivery provenance

- Parent branch：`feature/nonftv-phenotype-decodability-audit`
- Parent commit：`745282e77bb051dc0f8d9d41779691dcf8b307ce`
- Start timestamp：`2026-08-12T09:19:09-04:00`
- Experiment branch：`feature/bpe-source-observability-context-fov-audit`
- Audit commit SHA：`PENDING`
- Push status：`PENDING`

## 主要公开 artifacts

- Source contract：`reports/bpe_source_contract.md`
- Coverage：`metrics/coverage_table.csv`
- Boundary touch：`metrics/boundary_touch_table.csv`
- Physical margin：`metrics/physical_margin_distribution.csv`
- Laterality：`metrics/laterality_audit.csv`
- Longitudinal：`metrics/longitudinal_geometry.csv`
- FOV/cost：`metrics/acquisition_image_support_extent.csv`、
  `metrics/context_candidate_cost_table.csv`
- Figure：`figures/fov_comparison.png`
- Decision：`metrics/decision.json`
