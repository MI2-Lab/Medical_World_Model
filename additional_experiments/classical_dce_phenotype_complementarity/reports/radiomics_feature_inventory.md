# Classical DCE radiomics feature inventory

本报告由 `scripts/inventory.py` 从锁定的只读输入确定性生成。报告和两个 CSV 只含 schema 与 aggregate count，不含任何患者 ID 或 patient-level measurement。

## 1. 权威输入与 fail-closed 核验

| 输入 | 路径 | SHA-256 | 结构 |
|---|---|---|---|
| DCE measurement workbook | `/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx` | `f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc` | `datawith4visits`；384×29；used range `A1:AC385` |
| Clinical workbook | `/data/data/Breast_Cancer/I-SPY2/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` | `c016962d2d1e23686746ad3e74a58caeb2d1362f6393fd6209c10723f87c3a53` | `ISPY2_n985_TCIA_clinical`；985×10 |
| Complete-four-visit MRI fold manifest | `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv` | `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` | 808 unique patients；5 outer folds |

Radiomics workbook 只有一个 visible sheet；公式单元格 0，重复患者 0，重复整行 0。`CLINICAL-TRIAL-SUBJECT-ID` 为 384 个唯一、非缺失、恰好六位的数值 ID。匹配仅允许 workbook 六位 ID 与 canonical MRI ID 后缀精确等值，不做 fuzzy matching，也不在任何输出中物化 ID。

12 个 `*_pch_T0_Tk` 列均通过 `100 × (X_Tk − X_T0) / X_T0` 复算；全表最大绝对误差 `4.093e-12`。因此它们是派生字段，不是 12 个额外独立 measurement。

## 2. 实际 feature family

- **F / FTV**：functional tumor volume，单位 cc。
- **D / LD**：longest diameter；工作簿没有声明绝对单位。
- **S / SPH**：由 3-D FTV mask 得到的 sphericity，无量纲。
- **B / BPE**：对侧乳腺中央五层纤维腺体 mean early PE；工作簿没有独立声明绝对 scale/unit。
- **Other**：无。源表不是高维 PyRadiomics texture export。

绝对 measurement 共 16 列（4 family × T0–T3）；另有 12 列相对 T0 的 materialized percent change。`NONFTV = D + S + B`，`FULL = F + NONFTV`。

## 3. 逐列 inventory

| Column | Family | Visit | Unit | Role | Source missing | Workbook coverage | MRI-cohort coverage | Possible leakage concern |
|---|---|---|---|---|---|---|---|---|
| `CLINICAL-TRIAL-SUBJECT-ID` | ID | not_applicable | not_applicable | patient_identifier | 0/384 (0.00%) | 384/384 | 375/808 | Direct identifier: exact joins/splits only; never a predictor and never publish values. |
| `VOLUME_TUM_BLU_V10` | FTV | T0 | cc | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Pretreatment-only value; still fit every transform on outer train. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `VOLUME_TUM_BLU_V20` | FTV | T1 | cc | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `VOLUME_TUM_BLU_V30` | FTV | T2 | cc | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `VOLUME_TUM_BLU_V40` | FTV | T3 | cc | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `SPHERICITY_T0` | SPH | T0 | dimensionless | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Pretreatment-only value; still fit every transform on outer train. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `SPHERICITY_T1` | SPH | T1 | dimensionless | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `SPHERICITY_T2` | SPH | T2 | dimensionless | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `SPHERICITY_T3` | SPH | T3 | dimensionless | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `LD_T0` | LD | T0 | not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Pretreatment-only value; still fit every transform on outer train. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `LD_T1` | LD | T1 | not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `LD_T2` | LD | T2 | not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `LD_T3` | LD | T3 | not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `BPE_5slice_mean_T0` | BPE | T0 | native mean-PE scale; absolute unit not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Pretreatment-only value; still fit every transform on outer train. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `BPE_5slice_mean_T1` | BPE | T1 | native mean-PE scale; absolute unit not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `BPE_5slice_mean_T2` | BPE | T2 | native mean-PE scale; absolute unit not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `BPE_5slice_mean_T3` | BPE | T3 | native mean-PE scale; absolute unit not declared in workbook | absolute_measurement | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `FTV_pch_T0_T1` | FTV | T0→T1 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `FTV_pch_T0_T2` | FTV | T0→T2 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `FTV_pch_T0_T3` | FTV | T0→T3 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. FTV is tumor-burden/segmentation-derived and can be redundant with ROI mask geometry; fit residualization within outer train. |
| `Sphericity_pch_T0_T1` | SPH | T0→T1 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `Sphericity_pch_T0_T2` | SPH | T0→T2 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `Sphericity_pch_T0_T3` | SPH | T0→T3 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. SPH is derived from FTV-mask geometry and is not independent of FTV/ROI geometry. |
| `LD_pch_T0_T1` | LD | T0→T1 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `LD_pch_T0_T2` | LD | T0→T2 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `LD_pch_T0_T3` | LD | T0→T3 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. LD is another tumor-burden measure; later zero/floor values are valid observations, not missing values. |
| `BPE_pch_T0_T1` | BPE | T0→T1 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T1; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `BPE_pch_T0_T2` | BPE | T0→T2 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T2; using it earlier is future-visit leakage. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |
| `BPE_pch_T0_T3` | BPE | T0→T3 | % | derived_baseline_percent_change | 0/384 (0.00%) | 384/384 | 375/808 | Unavailable before T3; using it earlier is future-visit leakage. T3 is late/pre-surgery and must be labeled as such. Deterministic from T0 and the endpoint; do not count it as an independent measurement or duplicate a pipeline-derived change feature. BPE is contralateral/global; a lesion-centered MRI crop may not contain its source anatomy. |

完整机器可读版本：`features/radiomics_feature_inventory.csv`。

## 4. Missingness 与 matched population

- 源 workbook：384/384 patients 的 29 列全部非缺失；这是 workbook 内 complete-case，不代表完整 MRI cohort 无缺失。
- Clinical：全部 384 workbook patients 可精确匹配 clinical row；pCR、HR、HER2、MP 均 0 missing。
- MRI reference：375/808 (46.41%) 有 radiomics；433/808 (53.59%) 对全部 28 measurement 结构性缺失。另有 9 个 workbook patients 不属于 complete-four-visit MRI cohort。
- LD 零值数 T0/T1/T2/T3 为 0/6/65/128；BPE 为 0/0/1/4。这些是观察值，不得改写为 NA。

逐 scope、逐 column 的机器可读统计见 `metrics/missingness.csv`。其中 `complete4_mri_cohort` 明确把没有 source row 的患者记为 structural source unavailability，而不是在 384-row workbook 内伪造 NA。

## 5. Aggregate target coverage（384 primary population）

| Target | Valid | Missing | Positive | Positive rate |
|---|---|---|---|---|
| pCR | 384 | 0 | 113 | 29.43% |
| HR | 384 | 0 | 222 | 57.81% |
| HER2 | 384 | 0 | 90 | 23.44% |
| MP | 384 | 0 | 180 | 46.88% |

| HR/HER2 subtype | N | Rate |
|---|---|---|
| HR-/HER2- | 132 | 34.38% |
| HR-/HER2+ | 30 | 7.81% |
| HR+/HER2- | 162 | 42.19% |
| HR+/HER2+ | 60 | 15.62% |

Radiomics workbook 本身不含 outcome、molecular subtype、treatment 或 demographic 字段；这些 target aggregate 仅来自 SHA-locked clinical workbook 的 exact-ID intersection。

## 6. Timing-safe 使用边界

- T0 只能用 T0；T1 只能用 T0/T1；T2 只能用 T0–T2；T3 才能用 T0–T3，并必须标记为 `late/pre-surgery`。
- `*_pch_T0_Tk` 只有在 Tk 已观察后才可使用，并与其两个 endpoint 确定性重复。正式 pipeline 应从当时可见的 absolute value 按预注册公式重建 change。
- winsorization、log transform、scaling、imputation、feature selection 与 FTV residualization 全部只在 outer train 拟合。
- FTV 与 ROI/mask geometry 高度同源；SPH 直接依赖 FTV-mask geometry；LD 也是 burden proxy。不能把它们的增益直接解释成独立 biological signal。
- BPE 来自对侧/全乳背景组织。与 lesion-centered LOCAL latent 比较时必须注明 anatomy/input mismatch。
- 含 clinical label 的派生表不得使用“排除 label 的黑名单”；建模代码必须使用显式 predictor allowlist。

## 7. Population recommendation

Primary classical C/F/N/FULL、residualization、family ablation 与 HR/HER2 probe 使用全部 384 人；所有 paired model 在同一 complete population 上比较，结论不依赖 current MRI availability。MRI latent reference 与其 matched sensitivity analysis 使用严格 375 人 subset，并复用 MRI folds。二者必须分别标记为 `primary_384` 与 `mri_matched_375`，不得混报 AUROC。
