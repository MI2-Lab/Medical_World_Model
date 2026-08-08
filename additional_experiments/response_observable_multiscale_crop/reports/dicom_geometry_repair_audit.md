# DICOM 几何修复审计

## 范围与方法

本审计从私有 `patient_level_geometry.csv` 中筛出全部
`SFORM_SINGULAR` DCE visit，共 72 个、
19 名患者（T0=18、T1=18、T2=18、T3=18）。逐 series 仅读取 DICOM header：
`Rows/Columns`、`ImageOrientationPatient`、`PixelSpacing`、
`ImagePositionPatient`、`TemporalPositionIdentifier` 与 `AcquisitionTime`。
代码使用 `stop_before_pixels=True`，没有请求、解码或写出 PixelData。

空间 affine 由 DICOM LPS 坐标转换为 RAS+；IPP 沿 IOP 法向聚类、排序，
并以无序八角点 Hausdorff 距离比较 DICOM physical volume 与 mask sform。
中心角点与完整 voxel-footprint 角点均以
0.1 mm 为通过阈值。

## Header 审计结果

- 72/72 个 series 的文件数均等于 `X×Y×Z×T` 所隐含的
  `Z×T` cell 数；累计读取 77792 个 header。
- 72/72 的 Rows/Columns、IOP、PixelSpacing、IPP 层网格均完整一致；
  TPI 和 AcquisitionTime 两种分组均完整且一一对应。
- mask sform 与 raw DICOM physical volume 的最大中心角点 Hausdorff 误差为
  2.1859e-05 mm，低于 0.1 mm gate。
- 未出现 quarantine；header geometry gate 为
  **PASS**。

## Repair decision

| 决策 | visit 数 | 物理几何解释 | 当前可否进入模型 |
|---|---:|---|---|
| `TRUST_DCE_QFORM` | 35 | qform、mask sform、DICOM 一致 | 否 |
| `MASK_SFORM_GEOMETRY_CANDIDATE` | 37 | mask/DICOM 一致；DCE 不可靠 | 否 |
| `QUARANTINE` | 0 | header grid 或 affine gate 失败 | 否 |

37 个 mask-sform candidate 的 DCE qform–DICOM 角点误差为：
min=38.8774 mm、median=171.6 mm、
Q95=192.5 mm、max=197.5 mm。
这批数据不能用 DCE qform 或奇异 sform 作为物理真值。

35 个 `TRUST_DCE_QFORM` 仅表示 header-level physical geometry 可确认，
不表示 NIfTI 的 z/t pixel order 已经逐像素验收。为了与 Stage A 的 fail-closed
输入契约一致，72 个 singular-sform visit 均维持 `pixel_rebuild_required=true`。

## Model-ready gate

**Model-ready = false。** 本轮没有执行 DICOM pixel rebuild，也没有验证重建后
体素与 raw DICOM `(time, slice)` cell 的逐一对应；因此不能生成或使用
model-ready cache，Stage B 仍未授权。

解除 blocker 必须完成：从 raw DICOM PixelData 按时间组及 IPP 顺序重建全部
72 个 visit，应用像素缩放，写入经验证的 RAS affine，并验证每个 cell 恰好一次、
像素顺序正确、与 mask sform 的 physical corner 误差不超过 0.1 mm。

公开 CSV/JSON 仅含聚合计数、比例与误差；患者级明细保存在被 `.gitignore`
排除的本地 sidecar 中。
