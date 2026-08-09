# Single-Case Longitudinal Geometry Provenance Audit：预冻结计划

## 1. 唯一科学问题与当前状态

本实验只审计 C1B Model-Ready Pipeline 中唯一的 `ZERO_VALID_SOURCE_OVERLAP` visit：其异常是否存在一个完全由原始 DICOM/acquisition provenance 唯一决定、可程序化复现、与 outcome 无关的几何修复？公开材料中该病例只能称为 `CASE_ZERO_OVERLAP_001`。

本文件是 source-only、outcome-free 的预注册计划，不是审计结果。创建 scaffold 和运行合成 privacy test 不读取任何 private manifest、患者数据或 DICOM。既有 `c1b_model_ready_ftv_sanity` 结果保持 immutable：原 `STAGE_A=NO-GO` 与 `STAGE_B=NOT RUN` 永不追溯改写。

## 2. 冻结范围与禁止项

允许作为决策证据的只有 imaging metadata、raw PixelData、acquisition semantics、DICOM patient-coordinate semantics 和从 raw DICOM 独立重算的物理几何。候选选择不得读取或使用 lesion location、mask、bbox/ROI、FTV、LD、clinical、treatment、pCR、representation、loss 或 downstream model performance。

本轮禁止训练任何模型，禁止运行 Stage B，禁止修改 C1B crop contract、target grid、spacing、FOV、recenter、DCE7、模型、loss 或 population 的既有结果；禁止任意 flip、translation、rotation、registration sweep、deformable registration，以及为了让病例通过而进行 trial-and-error transform。图像注册只允许作 `DIAGNOSTIC ONLY`，永远不能成为正式 repair 或 series-selection 依据。

## 3. 病例定位与隔离

执行时，授权环境中的程序必须从既有 private failure manifest 精确定位唯一 `ZERO_VALID_SOURCE_OVERLAP` patient/visit，并验证命中数恰为 1；0 或大于 1 都 fail closed。真实 patient ID、study/series/frame UID、station name、source path、逐切片 metadata、raw coordinates 和 registration transform 只留在 gitignored private workspace，权限收紧，且不得打印到公开 stdout。

公开 Markdown、CSV、JSON 和 PNG 只允许匿名代号、相对公共路径、聚合/相对距离、角度、extent、overlap 和 categorical evidence。唯一允许发布的 center vector 是明确命名为 `source_center_ras_t0_relative_mm`、已逐 visit 减去 T0 center 的相对位移；绝对 RAS/LPS center、IPP、origin、affine translation 和 bbox corner 一律禁止。发布前必须运行 privacy gate；唯一允许的 case alias 是 `CASE_ZERO_OVERLAP_001`。

## 4. Phase A：完整 DICOM provenance inventory

对 failed visit 及同 patient 的 T0/T1/T2/T3，从 raw DICOM 读取并审计：PatientPosition、FrameOfReferenceUID、StudyInstanceUID、SeriesInstanceUID、SeriesDescription、ProtocolName、SequenceName、ImageType、Modality、BodyPartExamined、Laterality/ImageLaterality、Rows/Columns、PixelSpacing、SliceThickness、SpacingBetweenSlices、ImageOrientationPatient、ImagePositionPatient、InstanceNumber、TemporalPositionIdentifier、AcquisitionNumber/Time、ContentTime、TriggerTime、NumberOfTemporalPositions、MRAcquisitionType、ScanningSequence、SequenceVariant、TR/TE/FlipAngle、manufacturer/model、field strength、coil、table/position 和相关 reconstruction tags。SliceLocation 仅作辅助，绝不作为几何真值。

同 study 内枚举全部 series，并标注 native breast DCE、duplicate acquisition、derived/reconstructed、alternate phase stack、不同 acquisition plane、localizer/scout、subtraction、MIP、motion-corrected/registered series，以及 Spatial Registration 或其他 registration-related DICOM object。公开输出为匿名 provenance 摘要；逐实例证据保持 private。

## 5. Phase B：四访视 raw-DICOM physical geometry

每个 visit 均从 raw DICOM 独立重建 canonical DICOM LPS volume，再转换为 RAS；缓存坐标不能作为唯一真值。重算 source bounding box、oriented footprint、volume center、axis directions、physical FOV、extent 和 source spacing，并检查 affine finite/invertible、IOP orthonormality、IPP slice ordering/spacing、cell completeness、duplicate/missing、finite 和 nonconstant。

冻结比较包括 T0↔T1、T1↔T2、T2↔T3 及 failed visit↔T0：center displacement、orientation angle、axis-aligned和oriented overlap、minimum volume separation。该轨迹只用于区分 single-visit jump、持续 frame change 与 plausible repositioning，不反推 transform。公开表不含绝对 center、IPP、origin、affine 或 bounding-box corner；只可包含上述显式 T0-relative center displacement。

## 6. 最小 overlap translation：仅诊断

只计算平移量的 norm：使 C1B-H target grid 与 source acquisition 获得至少 1 voxel overlap，以及约 50%、90%、95% valid-source coverage 所需的最小 rigid translation。公开 `required_overlap_translation_summary` 只能报告距离和覆盖阈值，不报告 translation vector 或任何患者坐标。

这些量只回答偏移是毫米级 repositioning 还是几十/上百毫米的 coordinate mismatch；计算出的 translation 绝不能自动应用或成为 repair candidate。

## 7. Frame、position、orientation 与 protocol 审计

FrameOfReferenceUID 只分类为 same、different、missing/malformed。UID 不同本身既不表示错误，也不授权构造 transform；只有权威 DICOM registration relationship 或其他 source metadata 明确定义转换时才可使用。

比较 prone/supine、feet-first/head-first、左右 orientation、IOP sign pattern、IPP progression、acquisition plane、table orientation、scanner/site/protocol/coil/matrix/FOV/sequence fingerprint。scanner 或 protocol change 只作解释。只有证据证明当前 builder 违反标准 DICOM coordinate semantics，且修正规则唯一时，才可能进入 R1；若现有 LPS→RAS 已正确，禁止额外 flip。

## 8. Same-study series-selection gate

按既有、严格、outcome-free DCE eligibility contract 在 failed study 内重新搜索全部 series。候选必须是 original/primary、breast、native DCE，具有合法 temporal stack、geometry 和 sequence fingerprint；derived、subtraction、MIP、localizer 等全部排除。严禁使用“与 T0 overlap 最好”选择 series。

- 0 个合格候选：无 replacement。
- 恰好 1 个合格候选：仅成为待验证的 authoritative candidate。
- 大于 1 个同样合格候选：ambiguous，不得择一。

唯一 alternate candidate 必须完整执行 raw PixelData decode → temporal grouping → IPP ordering → scaling → affine construction → RAS canonicalization，并验证 cell completeness、duplicate/missing、finite、nonconstant、geometry 和 source overlap；只比较 header 不算验证完成。

## 9. Image-only registration：DIAGNOSTIC ONLY

允许对 failed visit→T0 做一次预注册的 image-only rigid diagnostic，记录 success/failure、translation norm、rotation angles、similarity 和 overlap，用来描述 extreme repositioning 与 physical-frame mismatch 的外观。其 transform 不得写入正式 geometry、不得选择 alternate series、不得改变 R1–R5 分类证据权重、不得救回病例。禁止 lesion/support-guided registration，禁止参数 sweep。

## 10. Root-cause 分类 R1–R5

- **R1 — AUTHORITATIVE_GEOMETRY_REPAIR**：发现明确 source metadata 错误或确定的历史 geometry conversion bug，且修正规则唯一、可程序化、可泛化、source-only，并通过独立 raw-DICOM 验证。
- **R2 — UNIQUE_VALID_ALTERNATE_ACQUISITION**：当前 series 不满足 source contract；同 study 恰有一个由 imaging semantics 独立选出的合法 native DCE replacement，且完整 raw pixel rebuild/geometry verification 通过。
- **R3 — TRUE/PLAUSIBLE_EXTREME_REPOSITIONING**：raw geometry 自洽，无 source 错误、无唯一 alternate，但 longitudinal physical acquisitions 确实不相交；这是 technical longitudinal eligibility failure，不用 registration 人工救回。
- **R4 — UNRESOLVED_COORDINATE_PROVENANCE**：metadata 无法区分真实 repositioning 与 upstream coordinate corruption，且没有唯一 source-authoritative correction；进入 technical eligibility failure。
- **R5 — AMBIGUOUS_REPAIR**：存在两个或更多同样合理的解释或修复，没有唯一权威依据；不得选择其中之一。

分类必须在读取任何 outcome、lesion 或 representation 结果之前完成并冻结。

## 11. Repair acceptance gate

只有 R1 或 R2 才有资格进入 repair，而且以下 10 项必须全部 PASS：

1. source-only；
2. outcome-free；
3. candidate unique；
4. evidence auditable；
5. rule programmatic；
6. raw PixelData independently verified；
7. no lesion-based choice；
8. no manual trial-and-error transform；
9. correction generalizable to every matching metadata pattern；
10. corrected valid-source-overlap 大于 0，且 corrected geometry 不破坏其他冻结 QC。

任一项 FAIL、UNKNOWN 或未执行均为 `NO REPAIR`。若 gate 全部通过，只能新建 repair candidate 和 repair validation，并建议以冻结规则启动全新的 Stage A；旧 Stage A 结论不变。

## 12. 最终 decision gate

- R1/R2 且 repair gate 全 PASS → `AUDIT-REPAIRABLE`；下一步是新的 Stage A with frozen repair rule。
- R3/R4 → `AUDIT-NOT-REPAIRABLE`；下一步是预注册 technical eligibility amendment。
- R5 → `AUDIT-AMBIGUOUS`；不得择一，下一步同样是 technical eligibility amendment。

若不允许 repair，下一轮在任何 representation 结果之前冻结：

`Eligible(patient) = AND over t in {T0,T1,T2,T3} [valid_source_voxels(patient,t) > 0]`

该 eligibility rule 不读 FTV、LD、clinical 或 pCR。`948 → 947` 只能作为待验证预期，正式 population 必须重新运行 eligibility 后确认，不能硬编码。

## 13. 冻结交付物

公开报告须为中文，并回答 raw geometry 自洽性、T0 距离和偏移量级、FrameOfReferenceUID、PatientPosition/IOP/IPP、唯一 alternate、确定性 builder bug、唯一 authoritative repair、diagnostic registration、outcome/lesion 非参与性、R1–R5、repair permission 和 amendment recommendation。

公开表固定为：四访视 metadata/geometry 摘要、pairwise geometry、same-study DCE semantic audit、root-cause evidence、repair acceptance gate。公开图固定为：T0–T3 physical box schematic、T0 对 failed footprint、visit center displacement、orientation angle difference、minimum separation；registration montage 和 alternate geometry comparison 仅在适用且已匿名时生成。PNG metadata 与可见/隐藏 byte strings 同样受 privacy gate 约束。

最终原则：**Source provenance decides repair. Outcome does not. Lesion does not. Model performance does not.**

## 14. Public privacy gate CLI

在 repository root、所有公开输出生成后执行并写入当前 gate：

```bash
python additional_experiments/zero_overlap_provenance_audit/scripts/audit_public_artifacts.py --overwrite
```

只做无副作用检查时执行：

```bash
python additional_experiments/zero_overlap_provenance_audit/scripts/audit_public_artifacts.py --check-only
```

任一 finding、无法读取/解析、畸形 PNG、public symlink 或未支持的 public 文件类型均返回非零；private 路径不打开，finding 也不回显匹配内容。
