# DICOM provenance inventory

## 结论

审计对象仅为 `CASE_ZERO_OVERLAP_001`；failed visit 为 `T3`。四访视均从 raw DICOM 独立重算几何并逐 cell 两次解码 PixelData，未把缓存坐标当作唯一真值。Patient ID、UID、source path、绝对 IPP/中心坐标、StationName 和 acquisition clock time 只存在于 gitignored private artifacts。

| visit | PatientPosition | raw orientation | matrix | spacing (mm) | FrameOfReferenceUID vs T0 | scanner | protocol group | raw geometry |
|---|---|---|---|---|---|---|---|---|
| T0 | FFP | PIR | [512, 512, 60] | [0.390625, 0.390625, 2.5] | SAME | GE MEDICAL SYSTEMS / GENESIS_SIGNA | PROTOCOL_GROUP_3 | PASS |
| T1 | FFP | LIP | [256, 256, 60] | [1.328125, 1.328125, 3.5] | DIFFERENT | GE MEDICAL SYSTEMS / GENESIS_SIGNA | PROTOCOL_GROUP_2 | PASS |
| T2 | FFP | PIR | [512, 512, 56] | [0.390625, 0.390625, 2.5] | DIFFERENT | GE MEDICAL SYSTEMS / SIGNA EXCITE | PROTOCOL_GROUP_1 | PASS |
| T3 | FFP | PIR | [512, 512, 56] | [0.390625, 0.390625, 2.5] | DIFFERENT | GE MEDICAL SYSTEMS / SIGNA EXCITE | PROTOCOL_GROUP_1 | PASS |

## Requested tag coverage

对每个 selected series 的所有 instance 读取了以下字段：`SOPClassUID, SOPInstanceUID, PatientPosition, FrameOfReferenceUID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription, ProtocolName, SequenceName, ImageType, Modality, BodyPartExamined, Laterality, ImageLaterality, Rows, Columns, PixelSpacing, SliceThickness, SpacingBetweenSlices, ImageOrientationPatient, ImagePositionPatient, SliceLocation, InstanceNumber, TemporalPositionIdentifier, AcquisitionNumber, AcquisitionTime, ContentTime, TriggerTime, NumberOfTemporalPositions, MRAcquisitionType, ScanningSequence, SequenceVariant, RepetitionTime, EchoTime, FlipAngle, Manufacturer, ManufacturerModelName, StationName, MagneticFieldStrength, ReceiveCoilName, TransmitCoilName, TableHeight, TableTraverse, TableMotion, TableVerticalIncrement, TableLongitudinalIncrement, TableLateralIncrement, TableAngle, TableType, TableSpeed, ReconstructionDiameter, ReconstructionMethod, ReconstructionAlgorithm, SeriesNumber`。可选字段中至少一个访视存在缺失的 tag：ImageLaterality, NumberOfTemporalPositions, ProtocolName, ReconstructionAlgorithm, ReconstructionMethod, SequenceName, StationName, TableAngle, TableHeight, TableLateralIncrement, TableLongitudinalIncrement, TableMotion, TableSpeed, TableTraverse, TableType, TableVerticalIncrement, TemporalPositionIdentifier, TransmitCoilName, TriggerTime。缺失 optional tag 不被补写为几何真值；`SliceLocation` 仅作辅助。

四访视 selected series 的 IOP 在各自 series 内一致且正交，PixelSpacing 一致，IPP 投影形成规则 slice grid，in-plane drift 通过冻结门限；TemporalPositionIdentifier / AcquisitionNumber / InstanceNumber phase-block 中由冻结 source contract 验证出的可用方法形成完整 temporal cell grid。公开文件不发布原始 IPP 或 acquisition time。

## Frame of Reference 与 PatientPosition

四访视 FrameOfReferenceUID 均存在，但属于四个不同等价类。同一个 patient 的不同 visit 使用不同 UID 并不自动代表错误，也不提供任何 transform。四个 selected series 的 PatientPosition 均为 `FFP`；该字段没有提供额外 flip/translation。DICOM 标准将 PatientPosition 定义为 annotation，而 ImagePositionPatient、ImageOrientationPatient 与 PixelSpacing 才定义 patient-coordinate image plane。因此现有 LPS→RAS builder 未因 PatientPosition 再施加一次翻转：[Patient Position](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.7.3.html)、[Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)。

## Same-study object 与 series inventory

failed study 可读 DICOM instances 为 505，SeriesInstanceUID 分组数为 6；Spatial Registration / Deformable Spatial Registration object 数为 0。因此没有 DICOM registration relationship 可将 failed frame 权威映射到 T0。Spatial Registration IOD 才是标准中描述不同 Frames of Reference 空间关系的对象：[Spatial Registration IOD](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.39.html)。

冻结、outcome-free 的 native-DCE contract 在 failed study 中得到 1 个严格候选，其中当前 selected acquisition 本身有效；排除当前 series 后 alternate candidate 数为 0。候选先依据 MR / ORIGINAL-PRIMARY / breast / native-DCE / temporal-stack / geometry semantics 决定，未读取 overlap、lesion 或 outcome。

## Scanner / protocol change

公开表只保留 manufacturer/model 与匿名 protocol group；StationName、free-text description/protocol 和日期时间留在 private inventory。`magnetic_field_strength_raw` 保留 legacy DICOM 原始编码，未作单位归一化，也未参与几何判定。T0/T1 与 T2/T3 存在 scanner-model/protocol 分组变化，可解释 acquisition context 改变，但不是自动构造坐标变换的依据。
