# ZERO_VALID_SOURCE_OVERLAP 单病例纵向几何 provenance audit

## 最终判定

**`AUDIT-NOT-REPAIRABLE`；root-cause class = `R4_UNRESOLVED_COORDINATE_PROVENANCE`；`REPAIR = NO`。**

`CASE_ZERO_OVERLAP_001` 的四访视 raw DICOM geometry 均在 series 内自洽，failed visit `T3` 的 selected acquisition 也通过 raw PixelData 双解码、temporal/slice cell 完整性、finite/nonconstant、IOP/IPP/spacing affine 与独立 rebuilt-NIfTI 对照。T0 与 `T3` source center 相距 **157.847 mm**，两个完整 source footprints 的最小间距为 **9.741 mm**，oriented physical intersection 为 **0.000 mm³**。

这不是几毫米级轻微 repositioning。虽然让 frozen C1B-H grid 首次出现 1 个 valid target voxel 的最小诊断平移为 13.191 mm，但达到 50% valid-source coverage 需要约 92.682 mm；90% 与 95% coverage 均因 source FOV 本身不足而无法仅靠 rigid translation 达到（最大可达 87.924%）。这些数值只描述失配尺度，**不是 repair transform**。

## 核心证据

- 相邻访视 source-center displacement：T0-T1=77.949 mm, T1-T2=59.130 mm, T2-T3=143.925 mm。观测上表现为末次访视 `T3` 的大跳跃；其后没有访视，无法检验该变化是否持续。
- T0、T1、T2、T3 的 FrameOfReferenceUID 均存在但互不相同；没有 Spatial Registration object 或 metadata-defined transform。UID 不同不能自行反推出 translation。
- PatientPosition 四访视均为 `FFP`；T0/T2/T3 raw orientation 一致，T1 的 acquisition plane 不同但与相邻访视仍有实质物理重叠。这证明当前 pipeline 已按 IOP/IPP 处理 plane change，不支持对 failed visit 再人工 flip。
- failed study 严格 native-DCE candidates = 1，当前 series 有效；alternate candidates = 0。不存在唯一 alternate acquisition。
- 没有发现当前 builder 违反 DICOM patient-coordinate semantics；raw affine、LPS→RAS、真实 RAS canonicalization 与冻结 rebuilt source 一致。
- image-only rigid registration 状态为 `PASS（仅诊断）`。optimizer candidate 的 source-center translation magnitude 为 179.018 mm、rotation magnitude 为 34.144°；before NCC 因零物理重叠不可定义，after NCC 为 0.4724，valid-overlap fraction 从 0.000% 到 67.669%。该结果不能区分真实 repositioning 与 upstream coordinate corruption，且 transform 从未进入 repair selection。

## 13 个问题逐项回答

1. **failed visit raw DICOM geometry 是否自洽？** 是。header grid、IOP/IPP、spacing、temporal cells 与双次 PixelData decode 均 PASS。
2. **T0 与 failed visit physical source 相距多少？** center displacement = 157.847 mm；minimum footprint separation = 9.741 mm；oriented overlap = 0。
3. **轻微 repositioning 还是 catastrophic mismatch？** 属于 catastrophic longitudinal coordinate mismatch：单访视中心跳跃超过百毫米，50% target coverage 需要约 92.682 mm 平移，90%/95% 连理论上都不可达。
4. **FrameOfReferenceUID 是否解释异常？** 否。四访视 UID 均不同且无注册对象；UID 只标识 frame，不定义跨 frame transform。
5. **PatientPosition / IOP / IPP 是否有明确异常？** PatientPosition 一致；IOP 正交且 series 内一致；IPP 形成规则 slice grid。没有 metadata-authoritative flip/translation 异常。
6. **是否存在同 study 唯一合法 alternate native DCE？** 否。严格候选只有当前 acquisition；alternate = 0。
7. **是否发现当前 builder 的确定性 geometry bug？** 否。DICOM LPS affine、LPS→RAS 与 canonicalization 复核一致。
8. **是否存在唯一 source-authoritative repair？** 否。
9. **image-only diagnostic registration 显示什么？** `PASS（仅诊断）`；它提出大幅 image-driven rigid candidate，但不能成为 source provenance，也不能可靠区分 repositioning 与 coordinate corruption。
10. **lesion/outcome 是否完全未参与选择？** 是。读取列表为空：FTV、LD、clinical、treatment、pCR 与 model performance 均未读取；T0 grid 使用 acquisition-center fallback。
11. **R1–R5 分类？** `R4_UNRESOLVED_COORDINATE_PROVENANCE`。raw geometry 自洽，但 metadata 无法唯一地区分 plausible extreme repositioning 与 upstream coordinate corruption。
12. **是否允许 repair？** 不允许；repair gate 的 source-authoritative/unique/corrected-verification 条件不成立。
13. **是否进入新的 four-visit eligibility amendment？** 是，建议新 run 在任何 representation 结果之前预注册：`Eligible(patient) = AND over t in {T0,T1,T2,T3} [valid_source_voxels(patient,t) > 0]`。本 audit 不执行 population rerun；`948 → 947` 只能作为待验证预期，不能写成确认结果。

## 科学边界与下一步

上一轮 `STAGE_A_NO_GO` 保持 immutable，Stage B 未运行，C1B crop contract 未修改。本轮没有训练模型、没有 registration sweep、没有尝试任意 flip/translation/recenter，也没有用 lesion 或结果倒推 transform。

下一步应建立新的、outcome-free technical eligibility amendment 并从 frozen population 重新开始一个全新 Stage A。旧 run 不能追溯改写为 GO。

**Source provenance decides repair. Outcome does not. Lesion does not. Model performance does not.**
