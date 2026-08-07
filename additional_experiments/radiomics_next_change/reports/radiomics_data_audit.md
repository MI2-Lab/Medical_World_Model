# Radiomics/Measurement 数据审计报告

审计日期：2026-08-06

## 1. 结论摘要

- 实际找到的源文件不是高维纹理 radiomics 表，而是一份纵向 MRI measurement 工作簿：`/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx`。它只有一个 sheet（`datawith4visits`），共 384 行、29 列、384 个唯一患者。
- 29 列由 1 个患者 ID、4 类 measurement 在 T0–T3 的 16 个绝对值，以及 4 类 measurement 从 T0 到 T1/T2/T3 的 12 个百分比变化组成。确认的四类 measurement 是 FTV、sphericity、LD 和 BPE；没有发现额外的高维纹理特征列。
- 源工作簿的 384 名患者在上述 29 列均无缺失，也没有重复患者或重复整行记录。与当前 808 人完整四访 MRI cohort 进行严格 ID 等值匹配后，375 人匹配、433 人无 radiomics/measurement，覆盖率为 46.41%；另有 9 名工作簿患者不在 808 人 cohort 中。
- 375 名配对患者均有 T0、T1、T2、T3 的四类 measurement，因此 T0→T1、T1→T2、T2→T3 各有 375 个完整 transition，共 1,125 个相邻 transition。可选的 T0→T2、T0→T3 也各可构建 375 个，但第一轮 M2 计划只把相邻变化作为正式训练目标。
- 数据足以支持“所有 MRI 患者参与影像损失，仅 375 名配对患者参与 masked radiomics auxiliary loss”的 M2 设计，但 complete-case 不是随机子样本：radiomics 可用组的 pCR 比例更低，基线 ROI 体素数更大，治疗类别和亚型构成也有差异。
- FTV 与 DCE8 第 8 通道 ROI mask 的几何信息高度重复。375 人×4 访视的只读核验中，FTV 与 mask 体素数的 Spearman 相关为 0.935；FTV 变化与 mask 体素数变化的三个相邻 transition 相关为 0.870、0.737、0.728。因此保留 mask 的模型必须称为“ROI 辅助的 image-only”，并同时报告去 mask 的严格 image-only 对照，不能把 FTV auxiliary gain 直接解释为学到了独立于几何的治疗响应。

本阶段没有发现阻止 M2 实现的 ID 或时间点问题；主要风险是 53.59% 的 cohort-level 结构性缺失、complete-case 选择偏差、候选五折清单的 provenance 不完整，以及 FTV/ROI geometry 的强同源性。

## 2. 文件、sheet 与审计范围

| 用途 | 只读文件 | 结构或状态 |
|---|---|---|
| 原始纵向 measurement | `/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx` | 132,946 bytes；SHA256 `f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc` |
| 原始 sheet | `datawith4visits` | 384 行×29 列；没有其他 sheet |
| 原始 clinical/pCR | `/data/data/Breast_Cancer/I-SPY2/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` | 985 人临床表；用于显式 ID 核验和描述性 complete-case 比较 |
| 808 人 MRI cohort | `/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv` | 808 人，均有 T0–T3 MRI |
| 预处理长表 | `/data/data/Preprocessed/I-SPY2/mri_nact_features_long.csv` | 1,536 行，即 384 人×4 measurement 访视 |
| 候选五折清单 | `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv` | 4,040 行、808 人、5 fold；SHA256 `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` |
| DCE8 只读缓存 | `/data/data/Preprocessed/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96` | 808 名 cohort 患者均有缓存；每人四访，8 通道中的第 8 通道为 binary ROI mask |

完整只读文件清单见 `data_audit/data_file_inventory.csv`。clean 分支期望的 `_corejepa_clean_dce8` 和 `corejepa_response_features.npz` 当前不存在；本审计没有创建、覆盖或改写任何共享数据或 manifest。

## 3. 原始表结构与逐列 schema

源表是宽表，没有单独名为 `timepoint`、`feature_name` 或 `measurement_value` 的列：时间点、特征名和数值角色编码在列名中。`CLINICAL-TRIAL-SUBJECT-ID` 是患者字段；其余绝对值列为 measurement value。预处理长表将同一信息展开为 `visit` 和四个数值字段 `tumor_volume_blu`、`sphericity`、`ld`、`bpe_5slice_mean`，但该长表是源工作簿的派生物，不是额外数据源。

| 角色 | 原始列名（全部列） | 读取类型 | 时间点/变化 | 单位状态 |
|---|---|---|---|---|
| 患者 ID | `CLINICAL-TRIAL-SUBJECT-ID` | `int64` | 不适用 | 不适用 |
| FTV 绝对值 | `VOLUME_TUM_BLU_V10`, `VOLUME_TUM_BLU_V20`, `VOLUME_TUM_BLU_V30`, `VOLUME_TUM_BLU_V40` | `float64` | T0、T1、T2、T3 | cc；依据随附 DICOM 数据说明，工作簿自身未单列单位 |
| Sphericity 绝对值 | `SPHERICITY_T0`, `SPHERICITY_T1`, `SPHERICITY_T2`, `SPHERICITY_T3` | `float64` | T0、T1、T2、T3 | 无量纲 |
| LD 绝对值 | `LD_T0`, `LD_T1`, `LD_T2`, `LD_T3` | `float64` | T0、T1、T2、T3 | 源工作簿与字段字典未明示，不作单位假设 |
| BPE 绝对值 | `BPE_5slice_mean_T0`, `BPE_5slice_mean_T1`, `BPE_5slice_mean_T2`, `BPE_5slice_mean_T3` | `float64` | T0、T1、T2、T3 | 源工作簿与字段字典未明示，按原始数值处理 |
| FTV 百分比变化 | `FTV_pch_T0_T1`, `FTV_pch_T0_T2`, `FTV_pch_T0_T3` | `float64` | T0→T1/T2/T3 | % |
| Sphericity 百分比变化 | `Sphericity_pch_T0_T1`, `Sphericity_pch_T0_T2`, `Sphericity_pch_T0_T3` | `float64` | T0→T1/T2/T3 | % |
| LD 百分比变化 | `LD_pch_T0_T1`, `LD_pch_T0_T2`, `LD_pch_T0_T3` | `float64` | T0→T1/T2/T3 | % |
| BPE 百分比变化 | `BPE_pch_T0_T1`, `BPE_pch_T0_T2`, `BPE_pch_T0_T3` | `float64` | T0→T1/T2/T3 | % |

FTV 的 cc 定义由 `/data/data/Breast_Cancer/I-SPY2/ACRIN-6698-ISPY2-DWI-and-DCE-MRI-Data-Descriptions_20210520.pdf` 中的 FTV DICOM 字段说明支持。该材料没有给出本工作簿 LD、BPE 列的明确单位，所以本报告和后续变换都不擅自假设 LD 为 cm/mm 或 BPE 为百分数。

逐列的非空数、唯一值数、分位数、极值、计划变换和 IQR 异常标记见 `data_audit/radiomics_schema.csv`。

## 4. ID 显式匹配审计

匹配没有使用模糊规则。具体流程如下：

1. 工作簿 ID 必须是恰好六位十进制 `CLINICAL-TRIAL-SUBJECT-ID`；384 个值均满足六位格式，未观察到需要恢复的前导零。
2. MRI `patient_id` 必须完整匹配 `^(?:ISPY2-|ACRIN-6698-)(\d{6})$`。808 个 ID 全部通过，其中 590 个为 `ISPY2-` 前缀、218 个为 `ACRIN-6698-` 前缀；未发现前后空格、大小写变体或非预期连字符。
3. 只提取上述正则中的六位 trial ID，并要求它与 `clinical_patient_id` 的六位字符串完全相等。808 人中不一致数为 0，归一化六位 ID 的重复数为 0。
4. 只有在工作簿六位 ID 与上述经双重验证的六位 ID 完全相等时才标记 `has_radiomics=true`。TCIA/ACRIN 前缀本身不被当作近似匹配依据。

结果为 375/808 人匹配（46.41%），433/808 人不匹配。工作簿中另有 9 人不属于 808 人完整四访 MRI cohort：`246134`、`495440`、`516763`、`652480`、`745633`、`748611`、`790722`、`889225`、`893874`。这 9 人的 measurement 表仍有 T0–T3，但 MRI 索引仅为三访，故不纳入当前 paired cohort。

逐患者的两套 ID、匹配规则、缓存路径、标签和五折位置见 `data_audit/radiomics_patient_overlap.csv`。该文件明确区分“未匹配”与“匹配”，没有为 433 人生成任何猜测映射。

## 5. 缺失、重复与访视完整性

- 原始工作簿：384×29 个单元格中的核心字段无缺失、无 NaN/Inf；384 个患者 ID 唯一，重复患者数 0，重复整行数 0。
- 派生长表：384 人各有 T0、T1、T2、T3 四行，共 1,536 行；`clinical_patient_id + visit` 重复数为 0。
- 808 人 MRI cohort：375 人具有完整 measurement，433 人在所有四类、所有访视上均为结构性缺失，单列 cohort-level 缺失率均为 433/808=53.59%。这不能用全队列均值进行简单单值填补。
- 375 人 paired subset：四个绝对 measurement 在 T0–T3 全部有效；三个相邻 transition 的每个 feature mask 均为 375/375 有效。
- 工作簿的 12 个 `*_pch_*` 列与由绝对值重新计算的 `(r_t-r_0)/r_0×100` 一致，最大绝对数值误差小于 `5×10^-12`。它们不是独立观测，后续不会与由绝对值生成的变化目标重复计入 loss。

详细缺失统计见 `data_audit/radiomics_missingness.csv`。M2 应保留 808 人的 image loss，仅在患者、transition、feature 均有效时计算 masked radiomics loss；不能因缺少 measurement 而删除 433 名 MRI 患者。

## 6. Transition 与五折数量

每个 matched patient 都有完整四访，因此三个相邻 transition 的有效人数相同。下表中“总 MRI/配对”给出每个 split 的患者数；括号内为该 split 的三个相邻 transition 合计数。

| Fold | Train：总 MRI/配对（3-transition） | Validation：总 MRI/配对（3-transition） | Test：总 MRI/配对（3-transition） |
|---:|---:|---:|---:|
| 0 | 525/247（741） | 121/59（177） | 162/69（207） |
| 1 | 525/239（717） | 121/69（207） | 162/67（201） |
| 2 | 525/240（720） | 121/52（156） | 162/83（249） |
| 3 | 526/242（726） | 121/61（183） | 161/72（216） |
| 4 | 526/225（675） | 121/66（198） | 161/84（252） |

对任一表格中的配对人数，T0→T1、T1→T2、T2→T3 分别都有同样数量的全特征有效样本；详细逐 transition 数量见 `data_audit/radiomics_transition_counts.csv`。全 cohort 的三个相邻 transition 各 375，共 1,125。

由于绝对值四访完整，如后续只用于 grounding，可额外构建 T0→T2 和 T0→T3，各 375 个；其每 fold/split 数量也等于表中配对人数。第一轮 M2 不把这两个长间隔 transition 混入训练，以免时间间隔差异成为额外混杂。原始相邻 target 明细见 `data_audit/radiomics_transition_targets_raw.csv`。

候选五折清单满足 808 人各恰好在一个 fold 的 test 集出现一次。不过它是从旧资产中找到的有效候选副本，当前 clean 分支没有配套的五折 checkpoint，也没有提交能够证明其原生生成链的 metadata。因此可以在 M0/M1/M2 中统一锁定使用，但报告必须写成“候选五折协议下的新训练”，不能宣称数值复现了 native clean 五折结果。

## 7. 数值范围、单位与异常检查

| 特征 | 四访总范围 | 零值/负值 | IQR 异常标记 | 审计解释 |
|---|---:|---:|---:|---|
| FTV | 0.01196–471.31446 | 0/0 | 各访视 30–51 个 | 严格为正但右偏明显；适合用 fold-train 拟合的 `epsilon` 做 log-change，并进行训练折限定的 robust scaling |
| Sphericity | 0.00829–0.79678 | 0/0 | 各访视 3–14 个 | 位于合理的非负有限区间；优先使用绝对差而非放大低基线的百分比差 |
| LD | 0–14.0 | 199/0 | 各访视 13–24 个 | 零值真实存在；单位未知，不能未经 `epsilon` 直接取 log，也不能进行单位换算 |
| BPE | 0–104.755 | 5/0 | 各访视 15–24 个 | 单位未知；零值存在，按原始数值和 fold-specific 稳健变换处理 |

所有绝对 measurement 都是有限非负数。IQR 规则标出的长尾值暂时只作为审计 flag，不等价于数据错误，不据此删除患者。值得注意的变化范围包括：FTV 的源表 T0→T1、T0→T2 百分比最大分别为 +1,485.9% 和 +1,991.1%；相邻绝对 FTV 变化范围在三个 transition 中分别为 -217.22～+112.99、-151.15～+79.81、-370.31～+71.11。这些长尾使“先全队列 winsorize/标准化”不合规，所有阈值和中心尺度必须只在每 fold 的 training patients 上拟合，再原样应用到 validation/test。

## 8. FTV 与 ROI mask/geometry 的高重复性

这不是纯统计巧合，而有明确的数据路径原因：legacy DCE8 的第 8 通道是 binary ROI mask；当前 `mask_geometry` 又从同一个 mask 确定性生成体素占比、三轴 bbox 尺寸、bbox 占比、填充率和中心位置等 9 维 geometry。FTV 也来自同一 I-SPY2 FTV 分析流程中的肿瘤区域与增强阈值。因此 FTV auxiliary target 与 mask/geometry 不是独立模态。

为量化重复程度，本审计只读加载 375 个 matched patients 的四访 per-visit DCE8 cache，共 1,500 个 patient-visit，不写回缓存。结果如下：

| 比较 | 样本数 | 相关性 |
|---|---:|---:|
| FTV 与 ROI mask 体素数，全部访视 | 1,500 | Spearman ρ=0.935 |
| `log1p(FTV)` 与 `log1p(mask体素数)`，全部访视 | 1,500 | Pearson r=0.848 |
| FTV 与 mask bbox 体积，全部访视 | 1,500 | Spearman ρ=0.658 |
| FTV 与 mask 体素数，T0/T1/T2/T3 | 每访视 375 | Spearman ρ=0.908/0.865/0.896/0.855 |
| FTV log-change 与 mask 体素数 log-change，T0→T1/T1→T2/T2→T3 | 每个 transition 375 | Spearman ρ=0.870/0.737/0.728 |

二者高度相关但不完全相同：1,500 个缓存 mask 中有 12 个为零体素（T1 3 个、T2 6 个、T3 3 个），对应工作簿 FTV 仍为正；paired subset 中没有全场全 1 mask。零 mask 不能直接解释为“无肿瘤”，需要在训练日志中单独标记并核对裁剪、ROI 来源或缓存构建路径。固定网格 mask 的体素数也不是 cc，不能拿它替代物理 FTV。

由此得到三个实验约束：

1. 保留第 8 通道 mask 的配置只能称为“ROI 辅助的 image-only”；正式结论必须同时给出去除第 8 通道和独立 geometry 的严格 7-channel 对照。
2. M2 radiomics head 只能读取 predicted image delta，不能读取 mask-derived 9-D geometry、clinical、treatment 或真实未来 measurement；否则无法判断 gain 是否来自影像变化。
3. 需要分别报告 FTV head 和其余 sphericity/LD/BPE head 的 grounding。若 gain 仅出现在 FTV 且去 mask 后消失，应解释为 geometry redundancy，而不是独立治疗响应表征。

## 9. Complete-case 偏差（仅描述性）

比较对象固定为 808 人完整四访 MRI cohort：radiomics 可用 375 人，不可用 433 人。以下分析不做显著性检验、不进行因果解释，也不使用 test outcome 调整训练策略或超参数。

### 9.1 主要差异

| 指标 | Radiomics 可用 | Radiomics 不可用 | 可用组减不可用组 |
|---|---:|---:|---:|
| pCR 比例 | 29.33%（110/375） | 38.11%（165/433） | -8.77 个百分点 |
| 年龄均值±标准差 | 48.26±10.14 岁（n=375） | 49.46±10.58 岁（n=430） | -1.20 岁 |
| MRI 访视数均值 | 4.00 | 4.00 | 0 |
| Radiomics 访视数均值 | 4.00 | 0.00 | +4.00 |
| MammaPrint `label_mp=1` | 46.67% | 48.04% | -1.37 个百分点 |
| T0 ROI mask 体素数中位数［IQR］ | 16,927［9,317，34,359］ | 12,991［6,151，27,167］ | +3,936（中位数） |

同定义的 baseline FTV 在 radiomics 不可用组不存在，所以不能完成 FTV-to-FTV 的基线 lesion volume 比较。上表最后一行是对 808 人 DCE8 第 8 通道的只读 ROI 体素数描述，仅作为固定网格几何 proxy；它不是物理体积，也不替代 FTV。可用组的 proxy 更大，提示 measurement 覆盖可能偏向基线 ROI 较大的患者。

### 9.2 HR/HER2 亚型与治疗类别

| 类别 | Radiomics 可用 | Radiomics 不可用 | 差值（百分点） |
|---|---:|---:|---:|
| HR+/HER2- | 42.13% | 37.41% | +4.72 |
| HR-/HER2- | 34.40% | 36.49% | -2.09 |
| HR+/HER2+ | 15.73% | 17.09% | -1.36 |
| HR-/HER2+ | 7.73% | 9.01% | -1.27 |
| targeted_other | 41.60% | 34.18% | +7.42 |
| her2_targeted | 28.00% | 31.41% | -3.41 |
| taxane | 16.80% | 18.48% | -1.68 |
| io | 7.20% | 7.85% | -0.65 |
| platinum_parp | 6.40% | 8.08% | -1.68 |

`treatment_family` 是审计用粗粒度归类，不是 radiomics 表字段，也不会作为 M0/M1/M2 的 transition 或正式 readout 输入。上述差异说明不能把 375 人的 auxiliary-supervised 子集当成 808 人的无偏随机样本。

### 9.3 Fold 分布

| Fold | Train 可用/不可用 | Validation 可用/不可用 | Test 可用/不可用 |
|---:|---:|---:|---:|
| 0 | 247/278 | 59/62 | 69/93 |
| 1 | 239/286 | 69/52 | 67/95 |
| 2 | 240/285 | 52/69 | 83/79 |
| 3 | 242/284 | 61/60 | 72/89 |
| 4 | 225/301 | 66/55 | 84/77 |

每个 fold 的 auxiliary-supervised 样本比例不同，尤其 fold 4 train 只有 225 人、fold 2 validation 只有 52 人。训练和超参数选择必须按 patient/fold 报告有效样本数，不能把五折重复出现的患者行相加后当作独立患者。

完整比例明细见 `data_audit/radiomics_complete_case_bias.csv`。

## 10. 泄漏防护与后续使用边界

- 原始绝对值和 `data_audit/radiomics_transition_targets_raw.csv` 只作为未标准化审计输入。每个 fold 的 log `epsilon`、winsorization 阈值、中位数、IQR、均值/标准差和任何 imputation 参数只能在该 fold 的 training patients 上拟合，并保存到独立的 `configs/radiomics_transform_fold_{k}.json`。
- Validation/test 只能应用 training-fold 参数；不得先在 375 人或 808 人全体上标准化。Test radiomics 不参与模型选择、loss 权重选择或 readout 拟合。
- Radiomics/measurement 只进入训练期 auxiliary loss 和训练后的 grounding 审计；正式 inference 与 frozen pCR readout 不得读取表格、真实未来 MRI 或真实未来 measurement。
- 所有 808 名 MRI 患者都保留 image loss。对 433 名 measurement 不可用患者，radiomics loss mask 为 0，而不是用 cohort 均值制造伪 target。
- 由于 FTV 与 ROI mask 高度重复，必须报告 8-channel ROI 辅助结果和 7-channel 严格 image-only 结果，并对 12 个零 mask patient-visit 做日志标记。
- Complete-case 比较使用 pCR 只是预先规定的描述性数据审计；观察到的 pCR 差异不会用于改 fold、删样本、调阈值或选择模型。

## 11. 审计产物与局限

本阶段生成或核验的机器可读产物为：

- `data_audit/radiomics_workbook_sheets.csv`：工作簿 sheet 和尺寸；
- `data_audit/radiomics_schema.csv`：29 列逐列类型、单位、缺失、分位数和异常 flag；
- `data_audit/radiomics_patient_overlap.csv`：808 人逐患者显式匹配、缓存和 fold 状态；
- `data_audit/radiomics_missingness.csv`：源表、808 人 cohort 和 paired transition 三个层级的缺失；
- `data_audit/radiomics_transition_counts.csv`：五折、split、相邻 transition 的有效数；
- `data_audit/radiomics_transition_targets_raw.csv`：1,125 个未变换相邻 target；
- `data_audit/radiomics_complete_case_bias.csv`：预先规定的描述性 complete-case 比较；
- `data_audit/data_file_inventory.csv`：数据资产清单；
- `data_audit/radiomics_audit_summary.json`：关键计数、checksum 和匹配规则摘要。

仍需保留的局限包括：LD/BPE 单位不明；9 名工作簿患者缺少当前 cohort 所需的完整 MRI；433/808 人无 measurement；FTV 与 mask/geometry 同源且高度相关；少量缓存 mask 为零而 FTV 为正；候选五折 manifest 缺少 clean 原生 checkpoint/provenance；当前审计只验证了数据可用性，不能预先证明 M2 会改善 image-derived pCR 或减少 copy-current。
