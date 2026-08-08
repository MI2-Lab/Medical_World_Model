# I-SPY2 Multi-feature MRI NACT 表结构审计

## 结论

真实读取的工作簿只有一个 sheet：`datawith4visits`，数据维度为 384×29（Excel used range `A1:AC385`）。384 行对应 384 个唯一六位 ClinicalTrialSubjectID；缺失单元格、无限值、重复患者和重复整行均为 0。工作簿 SHA-256 为 `f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc`。

表为一患者一行的 wide longitudinal structure，没有显式 `visit` 列。FTV 的原始列名特殊：`VOLUME_TUM_BLU_V10/V20/V30/V40` 分别对应 T0/T1/T2/T3；SPH、LD、BPE 直接在列名中编码 T0–T3。

## 工作簿结构核验

- Sheet names：`['datawith4visits']`；sheet state `visible`；无隐藏补充数据表。
- 行/列：表头 1 行 + 384 数据行，29 列。
- 合并单元格：0；公式单元格：0。
- 重复患者：0；重复整行：0；忽略 ID 后重复 measurement vector：0。
- 29 列全部 0% missing；ID 为 `int64`，其余 28 列为 `float64`。
- 工作簿不含 pCR、molecular subtype、treatment、age 或其他 clinical outcome/metadata 字段。

## 真正的 longitudinal measurement families

| Family | 原始四访列 | 角色 | 定义/单位 |
|---|---|---|---|
| FTV | `VOLUME_TUM_BLU_V10`–`V40` | reference target | 功能性肿瘤体积；cc |
| LD | `LD_T0`–`LD_T3` | formal candidate | 影像报告最长径；工作簿未明示单位 |
| SPH | `SPHERICITY_T0`–`T3` | formal candidate | 等体积球表面积 / 3D FTV mask 表面积；无量纲 |
| BPE | `BPE_5slice_mean_T0`–`T3` | formal candidate | 对侧乳腺中央 5 层纤维腺体 mean PE |

因此 formal candidate pool 恰为 LD、SPH、BPE；FTV 是 reference。论文方法用于补足工作簿未写出的 measurement 定义：https://pmc.ncbi.nlm.nih.gov/articles/PMC7695723/。

## 派生列验证

12 个 `*_pch_T0_Tk` 列均为已经物化的 baseline-relative 百分比，逐行验证公式 `100 × (X_Tk−X_T0)/X_T0`；各 family 最大绝对误差为 `{"FTV": 4.092726157978177e-12, "SPH": 4.973799150320701e-13, "LD": 4.547473508864641e-13, "BPE": 4.547473508864641e-13}`。它们不是新的独立 measurement，也不是 T1→T2/T2→T3 相邻变化，因此不进入 candidate pool；正式相邻 Δ 从四访 raw columns 重新计算。

## Coverage 与零值质量信号

| feature | workbook_total_patients | strict_mri_overlap_patients | T0_available_workbook | T1_available_workbook | T2_available_workbook | T3_available_workbook | complete_4visit_workbook | missing_pct_workbook |
|---|---|---|---|---|---|---|---|---|
| FTV | 384 | 375 | 384 | 384 | 384 | 384 | 384 | 0.000 |
| LD | 384 | 375 | 384 | 384 | 384 | 384 | 384 | 0.000 |
| SPH | 384 | 375 | 384 | 384 | 384 | 384 | 384 | 0.000 |
| BPE | 384 | 375 | 384 | 384 | 384 | 384 | 384 | 0.000 |

所有四类 measurement 在该 complete-case workbook 中均为 384/384 四访完整；这不能外推为原始 808 cohort 无缺失。零值不被改写为 NA：LD T0/T1/T2/T3 的零值数为 0/6/65/128，BPE 为 0/0/1/4。LD 后期存在明显 floor effect，可能包括病灶消失/不可测或编码下限；源表不能区分其语义，本 screening 保留并标记，不任意删除。

## Patient mapping

仅用 canonical patient ID 严格 regex 后缀、clinical ID 与 workbook 六位 ID 三者等值匹配，不做 fuzzy matching：workbook 384 人、MRI cohort 808 人、交集 375 人、workbook-only 9 人、MRI-only 433 人。workbook-only IDs：`246134, 495440, 516763, 652480, 745633, 748611, 790722, 889225, 893874`。

## 逐列 schema

| column | dtype | role | feature | visit_or_interval | n_missing | n_unique | n_zero | min | median | max |
|---|---|---|---|---|---|---|---|---|---|---|
| CLINICAL-TRIAL-SUBJECT-ID | int64 | patient_identifier |  |  | 0 | 384 | 0 | 100899.000 | 516023.500 | 999733.000 |
| VOLUME_TUM_BLU_V10 | float64 | reference_target | FTV | T0 | 0 | 384 | 0 | 0.996 | 15.589 | 433.379 |
| VOLUME_TUM_BLU_V20 | float64 | reference_target | FTV | T1 | 0 | 384 | 0 | 0.120 | 7.383 | 471.314 |
| VOLUME_TUM_BLU_V30 | float64 | reference_target | FTV | T2 | 0 | 384 | 0 | 0.019 | 1.767 | 377.382 |
| VOLUME_TUM_BLU_V40 | float64 | reference_target | FTV | T3 | 0 | 383 | 0 | 0.012 | 0.839 | 158.899 |
| SPHERICITY_T0 | float64 | formal_candidate | SPH | T0 | 0 | 384 | 0 | 0.053 | 0.198 | 0.602 |
| SPHERICITY_T1 | float64 | formal_candidate | SPH | T1 | 0 | 384 | 0 | 0.008 | 0.198 | 0.535 |
| SPHERICITY_T2 | float64 | formal_candidate | SPH | T2 | 0 | 384 | 0 | 0.009 | 0.229 | 0.797 |
| SPHERICITY_T3 | float64 | formal_candidate | SPH | T3 | 0 | 384 | 0 | 0.071 | 0.256 | 0.763 |
| LD_T0 | float64 | formal_candidate | LD | T0 | 0 | 88 | 0 | 0.800 | 3.700 | 13.200 |
| LD_T1 | float64 | formal_candidate | LD | T1 | 0 | 83 | 6 | 0.000 | 3.100 | 13.600 |
| LD_T2 | float64 | formal_candidate | LD | T2 | 0 | 72 | 65 | 0.000 | 2.100 | 14.000 |
| LD_T3 | float64 | formal_candidate | LD | T3 | 0 | 64 | 128 | 0.000 | 1.200 | 9.300 |
| BPE_5slice_mean_T0 | float64 | formal_candidate | BPE | T0 | 0 | 384 | 0 | 5.260 | 24.617 | 104.755 |
| BPE_5slice_mean_T1 | float64 | formal_candidate | BPE | T1 | 0 | 384 | 0 | 4.626 | 20.123 | 83.244 |
| BPE_5slice_mean_T2 | float64 | formal_candidate | BPE | T2 | 0 | 384 | 1 | 0.000 | 18.090 | 82.427 |
| BPE_5slice_mean_T3 | float64 | formal_candidate | BPE | T3 | 0 | 381 | 4 | 0.000 | 17.022 | 81.138 |
| FTV_pch_T0_T1 | float64 | derived_baseline_percent_change_non_candidate | FTV | T0→T1 | 0 | 384 | 0 | -98.661 | -46.454 | 1485.886 |
| FTV_pch_T0_T2 | float64 | derived_baseline_percent_change_non_candidate | FTV | T0→T2 | 0 | 384 | 0 | -99.695 | -88.132 | 1991.078 |
| FTV_pch_T0_T3 | float64 | derived_baseline_percent_change_non_candidate | FTV | T0→T3 | 0 | 384 | 0 | -99.902 | -94.577 | 321.145 |
| Sphericity_pch_T0_T1 | float64 | derived_baseline_percent_change_non_candidate | SPH | T0→T1 | 0 | 384 | 0 | -92.519 | -1.767 | 207.272 |
| Sphericity_pch_T0_T2 | float64 | derived_baseline_percent_change_non_candidate | SPH | T0→T2 | 0 | 384 | 0 | -92.186 | 9.476 | 431.611 |
| Sphericity_pch_T0_T3 | float64 | derived_baseline_percent_change_non_candidate | SPH | T0→T3 | 0 | 384 | 0 | -73.669 | 25.553 | 524.102 |
| LD_pch_T0_T1 | float64 | derived_baseline_percent_change_non_candidate | LD | T0→T1 | 0 | 245 | 47 | -100.000 | -14.286 | 482.353 |
| LD_pch_T0_T2 | float64 | derived_baseline_percent_change_non_candidate | LD | T0→T2 | 0 | 245 | 8 | -100.000 | -41.865 | 316.667 |
| LD_pch_T0_T3 | float64 | derived_baseline_percent_change_non_candidate | LD | T0→T3 | 0 | 208 | 4 | -100.000 | -67.054 | 261.538 |
| BPE_pch_T0_T1 | float64 | derived_baseline_percent_change_non_candidate | BPE | T0→T1 | 0 | 384 | 0 | -75.750 | -15.876 | 338.540 |
| BPE_pch_T0_T2 | float64 | derived_baseline_percent_change_non_candidate | BPE | T0→T2 | 0 | 384 | 0 | -100.000 | -23.818 | 147.839 |
| BPE_pch_T0_T3 | float64 | derived_baseline_percent_change_non_candidate | BPE | T0→T3 | 0 | 381 | 0 | -100.000 | -28.949 | 407.273 |
