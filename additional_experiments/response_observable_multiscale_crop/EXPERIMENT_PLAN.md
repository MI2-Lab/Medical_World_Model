# Response-Observable Multiscale Crop 实验计划

## 1. 科学问题与边界

当前固定 voxel crop 不能保证 FTV、LD、whole-lesion morphology 或 enhancement response target 能从 DCE7 图像中被观察。本实验先建立 physically consistent、lesion-complete、context-preserving、longitudinally coherent 的 image input contract，再讨论 representation learning。

Stage A 只研究输入，不训练模型。禁止 JEPA、FTV/LD grounding、pCR、clinical、treatment、transition、PCGrad、warm-up 与 foundation encoder。FTV support 只用于 localization、crop construction 与 outcome-free observability audit；mask、mask volume、bbox size、crop size、resize factor、FTV 和 LD 均不进入模型图像通道或 feature。

只有至少一个 causally deployable candidate 通过全部 Stage A gate 后，才允许另行执行最小 FTV-only representation sanity。即使 Stage A 通过，本任务也不自动执行 FTV+LD dual grounding。

## 2. 坐标与数据契约

- 原始 DCE/FTV support array 使用 NIfTI index `XYZ`；模型 tensor 使用 `CZYX`。
- `XYZ/CZYX`只定义数组轴顺序，不等于统一anatomical orientation；production必须另行冻结跨患者orientation canonicalization或验证native-orientation contract。
- 物理几何以 DCE acquisition grid 为准。优先使用经验证的 DCE sform；DCE sform 奇异时，只有 DCE/mask shape、spacing、orientation 与原始 DICOM 均一致，才允许复制同 index grid 的 mask sform 作为 header-only repair。repair 不读取 mask value。
- 每次 repair 或 quarantine 都写入 provenance。单纯 shape/pixdim 相等不等价于 affine registration。
- support voxel 使用完整 voxel footprint 做 containment，不只检查 voxel center。
- physical volume retention 在 source physical domain 计算；resampled mask 的 voxel count 不作为 retention 分母。
- 预注册的 crop containment 只判断 acquisition 内**可用 support**相对 crop 的保留。support 接触原始 image face 另记为 upstream acquisition-censoring sensitivity；不在看到结果后改写 primary metric，但它是完整 GO 的独立 blocker。
- LD 保持 source raw unit，仅做秩与 subgroup sanity；FTV support 不等同 radiologist target lesion。

## 3. 候选矩阵

| ID | 定位策略 | detail | context | 因果状态 |
|---|---|---|---|---|
| C0 | legacy T0 index-center projection | `32×96×96 voxel` | 无 | deployable reference |
| C1A | 当前 visit support center | fixed physical FOV，病灶超限时 expand | 无 | deployable，但有 temporal recenter 风险 |
| C1A-tight | 当前 visit bbox | bbox + fixed-mm margin 后直接 fixed-shape resize | 无 | distortion/size-normalization sensitivity，不推荐 |
| C1B | T0 physical center/frame | T0-only fixed physical FOV，T0 病灶超限时 expand | 无 | deployable primary candidate |
| C1C | T0–T3 union | union bbox + margin | 无 | `ORACLE-UNION / AUDIT ONLY` |
| C2A | 当前 visit center | visit-adaptive detail | visit-adaptive larger context | deployable，但两支均 recenter |
| C2B | T0 physical center/frame | anchored detail | anchored larger/lower-resolution context | deployable multiscale candidate |

## 4. Physical-space 策略

主策略 P1：先建立 physical window，再以 common target spacing 采样；nominal FOV 内保持固定 absolute scale，只有病灶超过 FOV 才 expand。配置中的 output shape 与 nominal FOV 决定固定有效 spacing。expand case 单独报告，不把 crop dimensions、resize factor 或 padding mask输入模型。

对照策略 P2：lesion bbox + fixed-mm margin 后直接 resize 到固定 tensor。它会让 occupancy 趋于恒定并弱化 absolute size；本轮只做 distortion/leakage sensitivity，不能因 containment 为 100% 就直接推荐。

margin 候选固定为 5、10、15、20 mm。选择规则不读取 pCR 或模型结果：先要求 available-support containment、large-LD 与 FTV gate 全部通过，再选使 nominal-FOV overflow/scale change 最少的最小 margin；并把四档完整结果留作敏感性。按该规则主分析选择 5 mm，10/15/20 mm 不因极小的 containment 增益而获得优先级。

## 5. Stage A 指标与 gate

延续 legacy 定义：

- `boundary_touch`：retained support 的 voxel footprint 接触任一 crop face；
- `suspected_truncation`：boundary touch、source-domain retention `<0.99` 或 support 不可审计；
- `severe_truncation`：retention `<0.90`；
- `sufficient_containment`：可审计、retention `>=0.99` 且不 touch；
- `exact_full_support_containment`：source support 每个 voxel footprint 均保留。

预注册 gate：

1. overall sufficient containment `>=95%`；
2. overall exact full-support retention `>=95%`；
3. visit-specific LD top-quartile suspected truncation `<=5%`，top-10% `<=10%`；
4. FTV retention Q05 `>=0.95`；
5. 无严重 anisotropic scaling/resampling distortion；
6. morphology surface retention 与 cut-component audit 支持未来监督；
7. 无明显 artificial longitudinal normalization；
8. 当前 timepoint inference 不需要 future mask/bbox；
9. 不直接输入 geometry target 或 crop-scale metadata。
10. raw DCE phase selection、完整 3-D DCE7 单次重采样、归一化与 cache round-trip 已验收。
11. 跨患者anatomical orientation contract已冻结并验收。

Large-LD primary gate 使用每个 visit、每个 view 的最坏 crop-specific rate。另报告把 source-face touch 计入的上游敏感性；若该敏感性越界，只能声称 available-support crop 子门通过，不能声称 end-to-end whole-lesion observability 已证实。

`C1C` 即使通过也不能触发 GO。若只有 audit-only candidate 通过，Decision 仍为 PARTIAL 或 NO-GO。

## 6. 必做分析

- spacing、shape、axis order、orientation、qform/sform、DCE-mask affine、跨 visit affine/spacing variation；
- current crop physical FOV 分布；
- C0/C1/C2 overall 与 T0–T3 containment；
- visit-specific LD top quartile/top 10%；
- FTV voxel/physical-volume/extent retention；
- surface retention、bbox containment、cut connected components；
- lesion 外 context margin 与 context/lesion physical-volume ratio；
- effective output spacing、XYZ resize factor、anisotropy、padding/expansion；
- crop center drift、lesion relative position、FOV/scale/context longitudinal variation；
- small/medium/large lesion 与一名 T0–T3 longitudinal image preview；
- crop 后 normalization 改变的代表性 image-quality sensitivity。

本轮代码实现的生产边界是 physical window、source-domain geometry audit 与真实二维中心层 preview；它没有生成可训练的完整 3-D DCE7 cache。二维 normalization sensitivity 和理论 tensor footprint 只能暴露风险，不能替代 model-input pipeline 验收。

## 7. Decision

- `INPUT-CONTRACT GO`：至少一个 deployable candidate 通过所有 gate；
- `INPUT-CONTRACT PARTIAL`：containment 显著改善，但 temporal normalization、geometry repair、distortion、context 或 deployment leakage 仍未解决；
- `INPUT-CONTRACT NO-GO`：adaptive/multiscale 仍不能可靠观察 lesion。

输出顺序保持为：`Observable Image → Grounded Response State → Response Dynamics → Clinical Readout`。
