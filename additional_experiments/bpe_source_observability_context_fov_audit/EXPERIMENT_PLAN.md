# BPE Source Observability & Context-FOV Audit

## 1. 目标与 estimand

本实验只回答一个 input-target observability 问题：BPE 的真实 source region 在当前
C1B-H / LOCAL input contract 下是否物理可见，以及最小因果有效 context FOV 是
LOCAL、完整 C1B-H，还是需要 bilateral acquisition context。

实验不训练 encoder、JEPA、phenotype head、pCR model、FTV+LD、attention、segmentation
或 region-aware architecture。任何 pCR、HR、HER2、treatment、BPE magnitude、model
performance 字段都不允许进入 population、FOV、case sampling、gate 或推荐逻辑。

## 2. 冻结 provenance

- parent branch：`feature/nonftv-phenotype-decodability-audit`
- parent commit：`745282e77bb051dc0f8d9d41779691dcf8b307ce`
- start timestamp：`2026-08-12T09:19:09-04:00`
- experiment branch：`feature/bpe-source-observability-context-fov-audit`
- source phenotype commit：`f49cf17237a95e9f8b99ad5f13c73f90e1a94a28`

旧实验目录只读；所有新增代码和公开结果仅写入本目录。

## 3. BPE source contract 的证据层级

真实 source semantics 只允许来自 source workbook 的冻结字段、Goal 6 inventory 和原始
方法论文。字段为 `BPE_5slice_mean_T0` 至 `BPE_5slice_mean_T3`；BPE 全称为
background parenchymal enhancement。方法定义为：每次 DCE examination 中，自动分割
对侧乳腺边界，以 fuzzy c-means 区分 fibroglandular tissue，并在 superior–inferior
方向几何居中的连续五个 axial slices 内取 fibroglandular tissue 的 mean early percent
enhancement。论文支持每次 examination 有独立 scalar measurement，但未说明真实 FGT
mask 是逐访视重分割、T0 传播还是纵向注册；因此 ROI temporal anchor 保持未知。

字段值、方法文字或 raw acquisition 不能替代生成该字段时的真实 breast/FGT mask、五层
selection 和 laterality mapping。若这些 source artifacts 不存在，必须输出
`SOURCE_ROI_NOT_AVAILABLE`，不得重跑或自创 segmentation 并冒充原 source ROI。

## 4. Population 与只读输入

Primary population 是 Goal B 中 BPE-complete、complete-four-visit C1B-H matched cohort。
脚本只读取：

- source workbook 的 trial ID 和四个 BPE 字段的 missingness/finite 状态；
- frozen transition table 的 patient ID、trial ID 和 `bpe_valid`；
- frozen C1B grid、source geometry、valid-source overlap、cache availability；
- frozen RAS orientation audit；
- raw DICOM patient-directory availability。

脚本不读取 clinical workbook、pCR、HR、HER2、treatment 或 BPE 数值用于任何 gate。
九个 workbook-only patients 的排除只能归为 `NO_COMPLETE_FOUR_VISIT_C1B_MATCH`。

## 5. 三个 FOV contracts

- **F0 — Current LOCAL support**：固定 `64 x 64 x 64 mm` readout support；不是独立
  model tensor，也不是重新选择的 crop。
- **F1 — Full C1B-H tensor support**：固定 `112 x 176 x 160` ZYX、
  `0.9 x 0.9 x 2.0 mm` XYZ、`144 x 158.4 x 224 mm` XYZ 的 lesion-centered/T0-anchored
  RAS+ physical grid；不得重心化。
- **F2 — Reconstructed source-image support**：现有 reconstructed DCE source affine
  定义的 voxel-footprint parallelepiped，并 hash-link 到 frozen inventory 中的 selected raw
  DICOM series。Raw series existence 与 first-instance geometry tags 会检查，但本 audit 不从
  full raw series 独立重建第二个 footprint，因此不声称完成 raw↔reconstructed footprint
  equivalence。公开 X/Y/Z extent 分位数只是各轴分别计算的 RAS AABB marginal summaries，
  不是一个可实现的 median tensor，也不是 exact F2 containment support 或 breast boundary。

## 6. Geometry implementation

复用冻结 C1B geometry 的 voxel-center convention：affine 映射 voxel center；physical
support 使用 `[-0.5, shape-0.5]` 的完整 voxel footprint。每个 source affine 必须 finite、
homogeneous、invertible；C1B affine/center/shape/spacing 必须和 frozen grid manifest 一致。
所有 source orientation 必须是经 array permutation/flip 后的 RAS+，不能只检查 header
label。

若真实 ROI 可用，coverage、boundary touch、centroid inclusion、physical margin 和
valid-source support 必须在 physical RAS geometry 中计算。当前运行若没有真实 ROI，以上
字段统一为 `NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE`；不得用 lesion/FTV mask、body mask、
whole acquisition 或人工构造的 contralateral half-volume 填充。

## 7. Gates 与分类优先级

可评估时依次应用：

1. F0：至少 99% visits occupancy >=0.99、至少 99% 无 boundary touch、零 zero-overlap；
2. F1：相同条件，另要求 source physical margin Q05 >=5 mm；
3. F2：仅在 F1 fail 后判断 source ROI 是否在 acquisition 内完整。

但 `source_roi_available=false` 时优先、唯一合法分类为
`D / BPE_SOURCE_NOT_RELIABLY_AUDITABLE`。此时 F0/F1/F2 gates 是 `NOT_EVALUABLE`，不是
PASS 或 FAIL；不得从 scalar BPE decodability、nominal FOV size 或解剖常识推断覆盖率。

## 8. Laterality、longitudinal 与 cost

20 个 case 由 `sha256("bpe-fov-audit-v1|" + patient_id)` 排序抽取，和 outcome、target
magnitude 无关。抽样只验证 frozen RAS/affine/grid consistency；若 lesion-side-to-
contralateral mapping 或 source ROI 不存在，source laterality 结论保持不可审计。

Longitudinal table 固定列出 T0→T1、T1→T2、T2→T3。没有 visit-specific source ROI
时，centroid displacement、volume variation 和 laterality consistency 均为 NA，不用
acquisition center 或 lesion mask 替代。

Cost audit 精确报告 F0/F1，并描述 F2 reconstructed source-image support 的 marginal 分布。
F0 只是 readout，
encoder 仍读取完整 F1，因此 F0 的实际 input memory 与 F1 相同；可另列 64-mm cube 在
C1B spacing 下的假设 voxel 数，但不得把它冒充 input cost。只有分类 A/B
才允许定义 CXT1/CXT2 production candidate；分类 D 不输出伪造的 bilateral tensor size。

## 9. Required outputs

- `reports/bpe_source_contract.md`
- `metrics/coverage_table.csv`
- `metrics/boundary_touch_table.csv`
- `metrics/physical_margin_distribution.csv`
- `metrics/laterality_audit.csv`
- `metrics/longitudinal_geometry.csv`
- `metrics/context_candidate_cost_table.csv`
- `figures/fov_comparison.png`
- `metrics/decision.json`
- `reports/final_report.md`（中文）

所有 patient-level sampling 和 geometry rows 只写入 gitignored、mode `0600` 的
`*.private.csv`；公开文件只有 aggregate counts/distributions，不含 patient ID、UID、
DICOM path、workbook 或 mask。

## 10. Reproduction and validation

在 repository root、具备 NumPy、pandas、pydicom、Matplotlib、Pillow、openpyxl 与 pytest
的既有项目环境中运行：

```bash
python additional_experiments/bpe_source_observability_context_fov_audit/scripts/run_audit.py
python additional_experiments/bpe_source_observability_context_fov_audit/scripts/generate_artifacts.py
python -m pytest -q additional_experiments/bpe_source_observability_context_fov_audit/tests
python additional_experiments/bpe_source_observability_context_fov_audit/scripts/validate_audit.py
```

`run_audit.py` 会在读取数据前验证所有冻结输入的 SHA-256；任一 source、cohort、geometry
或 cache contract 漂移都会 fail closed。`validate_audit.py` 再验证 scientific decision、NA
语义、outcome firewall、公开 artifact privacy 与 private sidecar 的 gitignore 状态。
