# C1B Model-Ready FTV Sanity 最终报告

## 最终判定

**`STAGE_A = NO-GO`；`STAGE_B = NOT RUN`。**

冻结的 model-input population 为 948 人（808 + 140）、3,792 visits。完整 source-overlap audit 在其中发现 1 个 `ZERO_VALID_SOURCE_OVERLAP` visit：目标 grid 的有效 source voxel 为 `0 / 3,153,920`。该失败占 1/3,792 visits（0.0264%）和 1/948 patients（0.1055%）；其余 visit 的最小非零 exact valid fraction 为 5.1013%。

这个病例使 production-like validation 在 cache worker 提交前 fail closed。263 人 validation set 中已有 262 个 schema-3 cache 原子完成，但 262/263 不能证明 cohort-level contract 完整，因此不得进入 full-scope cache 或 Stage B。冻结人口没有因失败被事后修改，阈值没有放宽，失败病例也没有被事后排除。

机器可读结论见 `STAGE_A_NO_GO.json` 和 `metrics/stage_a_model_ready_gate.json`。不存在 `STAGE_A_GO.json`。

## Stage A 结果

| QC 项 | 状态 | 主要观测 | 解释 |
|---|---:|---|---|
| 正式 singular-sform pixel rebuild | PASS | 72/72 visits；77,792/77,792 cells；max cell error = 0 | 真正从 raw DICOM PixelData 重建，而不是只修 affine |
| 全 model-input singular rebuild | PASS | 146/146 visits；153,112 cells | 包括额外 74 个 base-only visits |
| 严格 I-SPY1 source eligibility | PASS | 140/156 patients；604/624 visits | 仅用 imaging source、geometry、phase 和 pixel 证据；最终 population 为 808 + 140 |
| Anatomical orientation | PASS | 3,792/3,792 RAS+；round-trip max 8.04e-14 mm | array 与 affine 一起真实 permutation/flip，不是 header relabel |
| Registration sensitivity 与策略冻结 | PASS | R 成功 858/1,125（76.27%）；最终唯一冻结 H | 此 PASS 表示审计完成且 H 已冻结，不表示 R 的 success gate 通过 |
| C1B-H available-support containment | PASS | 97.8%（1,500 formal visits） | 高于约 95% 的冻结要求 |
| C1B-H FTV retention | PASS | Q05 = 1.000 | 高于 0.95 的冻结要求 |
| Grounding observability contract | PASS | observable 1,486/1,500；mask=0 为 14/1,500 | mask 只控制 grounding loss，不删 base training，也不进入 model tensor |
| Resampling 与 source overlap | **FAIL** | 1/3,792 visits 的 valid-source voxel = 0 | orientation 正确不等于 longitudinal physical frames 必然重叠 |
| DCE7/cache cohort contract | **FAIL** | 262/263 validation caches | channel semantics 已实现并单测，但全 cohort round-trip 未闭合 |
| Stage A 总门禁 | **FAIL** | 948 patients、3,792 visits | unresolved catastrophic resampling 即强制 NO-GO |

完整 Table 1 位于 `metrics/table1_model_ready_preprocessing_qc.csv`。需要注意，`manifests/model_input_inventory_summary.json` 中的 964 人、3,856 visits 是严格 I-SPY1 eligibility 之前的 source inventory，不是最终 model-input 分母。

### DICOM、orientation 与失败机制

正式 72 个 singular-sform visits 均通过两次独立 PixelData decode、逐 `(time, slice)` cell exact compare、逐文件 scaling、finite/nonconstant、qform/sform 和 physical-footprint 验收。正式 72 个 visit 的最大 footprint corner error 为 2.186e-05 mm。

最终 3,792 个 model-input visits 全部真实转换为 RAS+。转换前分布为 LAS 1,021、LIP 1、LPS 146、PIR 559、RPS 2,065；DCE-mask footprint corner 最大误差为 0.001836 mm。

零重叠 visit 的 raw-DICOM cell、DICOM LPS→RAS affine、重建 NIfTI affine 和 canonicalization 经独立复核相互一致。因此现有证据不支持把它解释为 array permutation、RAS conversion 或 affine 构造 bug；它是未解决的 longitudinal patient-frame/source-geometry 不兼容。仅凭 outcome-free header 不能可靠区分极端 repositioning 与上游 patient-coordinate 错误，更不能依据 lesion 或结果手工翻转、平移或 recenter。

### H 与 R

C1B-H 是 H/R 两个候选中更适合作为正式输入的策略，但“候选中更安全”不等于整个 C1B pipeline 已 model-ready。

C1B-R 的 rigid fits 仅 858/1,125 成功（76.27%，要求至少 95%），267 失败；17/1,125 为 catastrophic transforms（1.511%，要求不高于 1%）。把失败计为 nonworse=false 后，nonworse 率为 73.87%（要求至少 75%），padding delta Q95 为 +6.386 percentage points（要求不高于 +5 points）。此外，22/858 个成功 transform 出现 anatomy residual >5 mm 且 localization residual <2 mm 的 R-specific pattern，人工 montage 也在 high-transform stratum 显示灾难性错位。

R 的 exact containment 为 98.267%，略高于 H 的 97.8%，两者 FTV retention Q05 都为 1.000；这些局部收益不足以抵消 R 的硬门失败。因此 R 被拒绝，正式策略冻结为 H。随后发现的 H source-overlap blocker 则使整个 Stage A 仍为 NO-GO。

### Downsampling、observability 与 leakage

Formal FTV cohort 中 35/1,500 visits（2.33%）任一轴 downsampling factor >2；全冻结 population 为 597/3,792（15.74%）。所有需要 downsampling 的 volume 都按固定的 source-domain anti-alias 后 linear fixed-grid resampling 处置，没有为 outlier 扩大 tensor 或改变 patient-specific scale。这个策略不能修复物理视野完全不相交，因此 resampling 硬门仍然失败。

Grounding observability mask 覆盖全部 1,500 formal visits：1,486 个 observable，14 个 mask=0，涉及 12 人；T0/T1/T2/T3 分别为 1/8/3/2 个不可观察 visit。它只决定 FTV grounding loss 是否计入，不过滤 base objective。

Grid anchor 只使用 T0 information。Future support 不参与 grid、tensor 或 selection；formal follow-up support 只作事后 QC，base-only future support 不读取。Clinical、treatment、pCR、LD 和 outcome 字段没有进入 input construction。Geometry、transform、valid-source mask、phase provenance 和 patient identifiers 均留在 sidecar，不进入 `[4,7,112,176,160]` model image tensor。

## 13 个问题的逐项回答

1. **72 个 singular-sform visits 是否真正 pixel-level 修复？** 是。72/72 和 77,792/77,792 raw cells 均通过独立逐 cell exact 验证，max cell error=0；另有 74/74 个 base-only singular visits 同样通过，总计 146/146。

2. **全部 MRI 是否统一到明确 orientation？** 是，针对冻结的 Stage-A population，3,792/3,792 visits 已真实重排到 RAS+。但 orientation PASS 只证明轴向约定和 array/affine 一致，不证明不同 visit 的 longitudinal physical frames 一定相交；后者正是本轮 blocker。

3. **C1B-H 还是 C1B-R 更适合作为 world-model input？** C1B-H。R 未通过 success、catastrophic、nonworse、padding、residual 和人工复核门，因此被拒；H 是唯一冻结候选，但受零重叠 visit 影响，当前仍不能用于 Stage B。

4. **C1B 是否真正 model-ready？** 否。1/3,792 visits 有零 valid-source voxel，且 validation cache 仅 262/263，完整 schema-3 cohort contract 未闭合。

5. **Source-edge observability mask 覆盖多少 visits？** 全部 1,500 formal visits 都有定义；1,486/1,500（99.07%）可用于 grounding，14/1,500（0.93%）mask=0。

6. **N1 是否优于 L1？** 未评估；Stage B 未运行。

7. **N3 是否优于 N1？** 未评估。

8. **N3−N1 是否比 L3−L1 更强？** 未评估；没有 Difference-in-Differences 结果。

9. **最明显改善发生在 Spearman 还是 natural-scale R²？** 未评估，不能从 Stage-A geometry QC 推断 representation endpoint。

10. **Observed ΔFTV 是否改善？** 未评估。

11. **Base optimization safety 是否改善？** 未评估；没有 checkpoint、training curve、base degradation 或 representation-std 结果。

12. **Legacy partial crop 是否可以解释之前 G3 的部分不稳定性？** 不能在本轮建立因果归因。Legacy partial crop 仍是已确认的 preprocessing/evidence-target mismatch，也是合理机制假设；但本轮没有 L/N 训练、representation 或 optimization endpoint，因此不能声称它已经解释 G3 instability。

13. **下一步应进入 FTV+LD、先解决 JEPA-grounding optimization，还是升级 encoder？** 当前三个选项都不是立即下一步。首先应以 outcome-free 方式修复或查清 longitudinal physical-frame/source-overlap 异常；若没有权威信息可修复，则必须先显式预注册技术排除或 population amendment，再从 Stage A 起完整重跑。只有 Stage A GO 后，才执行 FTV-only L1/L3/N1/N3 2×2；再根据 representation 与 optimization safety 结果决定是否优先优化 grounding 或升级 encoder。本轮不得直接进入 FTV+LD。

## Stage B 与科学分类

Stage B 没有被授权或运行。`checkpoints/`、`features/`、`predictions/` 和 `logs/` 除占位文件外没有实验产物。因此：

- Table 2（Static FTV）、Table 3（Observed ΔFTV）、Table 4（Optimization safety）和 Table 5（Difference-in-Differences）按硬门禁设计不生成，不是意外遗漏。
- Figures 7–14 同样按设计不生成。
- Figures 1–6 已生成，依次为 raw-DICOM repair、orientation、registration transforms、H-vs-R representatives、FTV retention 和 resampling factors。
- 不能在 `OBSERVABILITY-SUPPORTED`、`OBSERVABILITY + OPTIMIZATION BOTTLENECK`、`INPUT FIX NOT REPRESENTATION-LIMITING` 三种 Stage-B 分类中选择任何一种；正确表述只能是 `STAGE_A NO-GO / STAGE_B NOT RUN`。

## 后续合规路径

当前 run 不能通过事后删除唯一失败病例改成 GO。可接受的下一轮路径只有：

1. 从权威 raw acquisition/geometry provenance 得到 outcome-free、可审计的纵向 coordinate 修复，然后重跑完整 Stage A；或
2. 若客观无法修复，先建立新的、显式预注册的全队列技术 eligibility rule，例如要求四个 visits 均至少有一个 valid-source voxel。该 amendment 预计得到 947 人的新 population，但必须视为新 run，不能追溯挽救本轮结果。

不允许基于 lesion、FTV、clinical、treatment、pCR 或训练表现决定翻转、平移、排除或 FOV；也不允许为进入训练而降低硬门阈值。

## 交付与验证

- 唯一门禁 sentinel：`STAGE_A_NO_GO.json`。
- 完整 preprocessing Table 1：`metrics/table1_model_ready_preprocessing_qc.csv`。
- Model-input failure gate：`metrics/model_input_pipeline_h_validation_gate.json`。
- Stage-A 报告：`reports/stage_a_gate_report.md`。
- DCE7 contract 状态：`reports/dce7_phase_contract.md`，明确 semantics 单测通过但 cohort cache contract 未完成。
- 公开工件 privacy gate：`metrics/public_artifact_privacy_gate.json`。
- 代码验证：62/62 tests PASS，`py_compile` PASS，`git diff --check` PASS。
- 可复现 fail-closed 验证：preflight 退出码 1，262 个既有 cache 的 filename/size/mtime state 未改变，未提交 cache worker。

## 公开影像归属

Figure 04 及其 descriptive alias 是从 The Cancer Imaging Archive（TCIA）公开乳腺 MRI 数据制作的去标识化派生技术复核图。来源为 Li et al. (2022), *I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)*, DOI `10.7937/TCIA.D8Z0-9T85`，以及 Newitt et al. (2021), *ACRIN 6698/I-SPY2 Breast DWI*, DOI `10.7937/TCIA.KK02-6D95`。源 collection 采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本项目执行了 DICOM reconstruction、canonical reorientation、fixed-grid resampling、intensity normalization、anonymous size-stratum selection、registration comparison、crop、composite、difference-map coloring 和技术标注。修改由本项目完成，不代表 TCIA 或原数据作者背书。完整说明见 `figures/README.md`。

病例标识、源路径、坐标范围和逐病例证据只保存在 private artifacts。公开报告只给出聚合计数和 SHA-256 provenance。
