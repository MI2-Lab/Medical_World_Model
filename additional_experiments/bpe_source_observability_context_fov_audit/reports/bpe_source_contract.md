# BPE source contract

## 结论

本实验确认 BPE 的全称是 **background parenchymal enhancement（背景实质增强）**。
真实字段是 `BPE_5slice_mean_T0` 至 `BPE_5slice_mean_T3`。其 source semantics 可以由
原始方法论文和冻结 feature inventory 恢复，但生成这些标量时的真实 breast/FGT mask、
五层坐标与 patient-specific laterality mapping 没有随 workbook 或现有 manifests 发布。

**Source status：`SOURCE_ROI_NOT_AVAILABLE`。** 本实验没有、也不得用新分割或半乳 volume
伪造 historical source ROI。

## BPE_source_semantics

| 项目 | 冻结定义 | 证据状态 |
|---|---|---|
| 全称 | background parenchymal enhancement | confirmed |
| 乳腺侧 | 原发肿瘤/病灶的对侧乳腺（contralateral） | method confirmed；patient-level L/R mapping unavailable |
| slice | superior–inferior 方向几何居中的连续 5 个 axial slices | method confirmed；真实 indices unavailable |
| tissue | 自动分割乳腺中的 fibroglandular tissue（FGT） | method confirmed；真实 mask unavailable |
| intensity | early percent enhancement：precontrast 到 early postcontrast；不是 late PE 或 SER | method confirmed |
| statistic | 五层 FGT voxels 的 mean early PE | method confirmed |
| lesion mask | 不用于定义/测量 BPE voxels | method confirmed |
| lesion laterality | 选择 contralateral side 所必需 | required but mapping unavailable |
| ROI method | automatic breast-boundary segmentation + fuzzy c-means FGT classification | algorithmic, not manual |
| timing | T0、T1、T2、T3 各有 visit-specific scalar | confirmed |
| ROI temporal anchor | 未说明每次重分割、T0 propagation 或 longitudinal registration | not documented |
| workbook unit | 字段名/方法是 mean PE；workbook 本身没有独立 unit metadata | qualified |

方法上的 early PE 为 `(S_early - S_pre) / S_pre × 100%`。原论文将 early acquisition
描述为注射后约 2.5 分钟；late acquisition 参与 SER，但不属于这个 BPE 标量。

## BPE_source_geometry_requirements

若要可靠重跑 F0/F1/F2 observability gates，至少必须取得每个 patient × visit 的：

1. 生成 workbook BPE 时的真实 contralateral breast boundary；
2. 真实 five-slice indices 或对应 physical slab；
3. 真实 FGT voxel mask/contours，而不是后验重建的 proxy；
4. lesion side 与 contralateral side 的 hash-bound mapping；
5. source image/phase UID、DICOM LPS geometry、NIfTI/RAS affine；
6. segmentation 是否在 image boundary 被截断，以及需要的 pre/early phase common support；
7. ROI 是逐访视重分割、baseline anchored，还是通过何种 registration 传播。

映射必须沿 `raw DICOM LPS → reconstructed NIfTI → true array-reordered RAS+ → frozen
C1B-H grid` 执行，并以 voxel-center affine 和完整 half-voxel footprint 计算 physical
intersection。不能只算 array-index overlap；reflect padding 也不能算 valid source。

## 已恢复与未恢复的 evidence

- Source workbook：384 人，四个 absolute BPE 字段均 finite；workbook 只有一个 ID、16 个
  absolute scalars 和 12 个 derived changes，没有 mask、坐标、laterality 或 affine。
- Matched audit cohort：375 人 × 4 visits；另外 9 个 workbook patients 因没有完整四访视
  C1B match 被技术性排除。
- Raw/reconstructed support：1,500/1,500 selected DICOM series 与 reconstructed DCE 可用；
  375/375 frozen C1B caches 可用。F2 extent 取自 reconstructed source affine；本 audit
  不从 full raw series 独立重算 footprint equivalence。
- Existing masks 是 lesion/FTV analysis masks，不能替代 contralateral FGT source ROI。
- 先前 FOV firewall 也把 ROI/laterality mapping、overlap 和 boundary audit 记为 unavailable。

## Evidence provenance

- 方法论文：[Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI](https://doi.org/10.1038/s41523-020-00203-7)
- Open-access full text：[PMC7695723](https://pmc.ncbi.nlm.nih.gov/articles/PMC7695723/)
- 冻结 target inventory：`additional_experiments/nonftv_phenotype_decodability_audit/manifests/target_contract.csv`
- 冻结 FOV firewall：`additional_experiments/nonftv_phenotype_decodability_audit/metrics/bpe_fov_observability_audit.csv`
- Goal 6 feature inventory commit：`f49cf17237a95e9f8b99ad5f13c73f90e1a94a28`
- Source workbook SHA-256：`f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc`
