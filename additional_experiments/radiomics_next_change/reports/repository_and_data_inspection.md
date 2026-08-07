# 仓库、环境与数据检查报告

检查日期：2026-08-06
仓库：`$REPO_ROOT`
检查方式：仅只读检查 Git、源码、配置、Python/CUDA 环境、共享数据目录及本实验的数据审计产物；未覆盖原始数据、fold manifest、checkpoint 或现有结果。

## 一、结论摘要

1. 当前工作树位于目标分支 `feature/ispy-clean-corejepa`，commit 为 `c413ec86af04795434bdc19e65bbb006c966f379`。检查时没有 tracked/staged 改动；存在未跟踪的 `additional_experiments/` 与 `shortcut_audit/`，均予以保留。
2. clean 实现的当前模型不是 image-only：MRI latent 显式叠加 9 维 ROI geometry，image transition 显式接收临床/治疗 condition，Factorized Response State（FRS）还以 geometry 和 condition 产生 latent correction；正式 FLR 只读取该 FRS，而不是 image latent。
3. 当前任务所需的原始纵向 measurement 文件已定位。`Multi-feature-MRI-NACT-Data.xlsx` 有 384 人、29 列，包含四次访视的 FTV、sphericity、LD、BPE 及从 T0 计算的百分比变化；与 808 人完整四访 MRI cohort 可严格匹配 375 人，共有 1,125 个完整相邻 transition。
4. 仓库根目录和工作目录下没有 `data/`。clean 配置实际指向共享只读数据 `/data/data/Preprocessed/I-SPY2` 与 `/data/data/Preprocessed/I-SPY1`，因此本次递归定位是在这些真实配置路径及 `/data/data/Breast_Cancer/I-SPY2` 完成的。
5. clean 配置期望的 `/data/data/Preprocessed/I-SPY2/_corejepa_clean_dce8` 和 `corejepa_response_features.npz` 均不存在；`runs/corejepa_clean`、clean checkpoint、history、frozen state 和 FLR 结果也不存在。因此没有可直接数值复现的 clean baseline checkpoint。
6. 存在一个可只读复用的 68 GiB legacy DCE8 cache，覆盖 808 名 I-SPY2 与 156 名 I-SPY1 患者，但它使用 key `x` 和另一套文件命名，不能由 clean dataset 原样加载，需要实验 wrapper。
7. 存在结构有效的 seed-2026 候选五折 manifest：808 人、5 折、每人恰好一次 test；但该文件不被当前 clean 分支引用，旁边没有生成脚本、完整配置或配套 clean checkpoint。它只能作为“候选锁定副本”，不能据此宣称 native five-fold reproduction。

## 二、Git 与仓库状态

| 项目 | 检查结果 |
|---|---|
| 当前分支 | `feature/ispy-clean-corejepa` |
| 当前 commit | `c413ec86af04795434bdc19e65bbb006c966f379` |
| tracked/staged 修改 | 检查时未发现 |
| 未跟踪路径 | `additional_experiments/`、`shortcut_audit/` |
| 保护措施 | 未执行 reset、checkout、clean、删除或移动；未改写任何原始 manifest |

仓库顶层相关内容如下：

| 路径 | 实际用途 | 与本任务的关系 |
|---|---|---|
| `ispy_jepa_tmi_clean/` | clean CoRe-JEPA 实现、I-SPY 预处理、训练、FLR 与文档 | 当前 baseline 的权威代码位置 |
| `scripts/` | 两个 HCC-TACE 数据处理脚本 | 与 I-SPY Next-Change 训练无直接关系 |
| `datasets/` | HCC-TACE 等数据说明与小型清单 | 未找到 I-SPY2 fold 或 radiomics 训练数据 |
| `requirements.txt` | 仓库级数据处理依赖 pin | 与当前 `bowen` 环境存在版本漂移，见下一节 |
| `shortcut_audit/` | 既有、未跟踪的 shortcut 审计工作 | 保留，不覆盖 |

clean `README.md` 明确说明其模块名和 checkpoint schema 与旧 experimental monolith 不同，legacy development checkpoint 不能按名称直接载入；因此“架构对应”不能替代当前分支上的数值 checkpoint 复现。

## 三、运行环境与依赖

### 3.1 运行时

| 项目 | 实测值 |
|---|---|
| conda 环境 | `bowen` |
| 环境路径 | `$CONDA_PREFIX`（环境名 `bowen`） |
| Python | 3.11.14 |
| PyTorch | 2.9.1+cu130 |
| PyTorch CUDA runtime | 13.0 |
| cuDNN | 9.13.0（API 返回 `91300`） |
| CUDA 可用 | 是 |
| GPU | 3 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| 单卡显存 | 97,887 MiB |
| NVIDIA driver | 580.76.05 |

### 3.2 主要 Python 包

| 包 | `bowen` 实测版本 | 仓库声明/备注 |
|---|---:|---|
| numpy | 2.2.6 | 根 `requirements.txt` pin 2.0.2；clean `pyproject.toml` 要求 ≥1.24 |
| pandas | 2.3.3 | 根文件 pin 2.2.2；clean 要求 ≥2.0 |
| scipy | 1.16.3 | clean 要求 ≥1.10 |
| scikit-learn | 1.8.0 | clean 要求 ≥1.3 |
| pydicom | 3.0.2 | 根文件 pin 2.4.4；clean 要求 ≥2.4 |
| SimpleITK | 2.5.3 | 根文件 pin 2.5.4 |
| PyYAML | 6.0.3 | clean 要求 ≥6.0 |
| tqdm | 4.67.1 | 根文件 pin 4.66.5；clean 要求 ≥4.66 |
| openpyxl | 3.1.5 | 满足 Excel optional dependency |
| matplotlib | 3.10.8 | 可用于图表 |
| seaborn | 0.13.2 | 可用于图表 |
| nibabel | 5.3.3 | 可读取 NIfTI |
| pytest | 未安装 | clean 的 test optional dependency 要求 ≥8.0；正式测试前需处理 |

结论：核心训练、Excel 审计与 GPU 运行依赖可用，但环境并非根 `requirements.txt` 的精确 pin 复现。后续结果必须保存实际版本信息，不能只引用 requirements。`pytest` 缺失不阻塞训练脚本的静态检查，但会阻塞 clean 自带 pytest 测试入口。

## 四、第一阶段要求的 15 项代码检查

### 4.1 总览

| 编号 | 检查项 | 代码位置 | 实际实现与结论 |
|---:|---|---|---|
| 1 | 当前 CoRe-WM 训练入口 | `ispy_jepa_tmi_clean/scripts/pretrain.py`；`corejepa/training/runner.py` | CLI 读取 YAML 后调用 `train()`。默认配置为 `configs/paper_v1.yaml`；训练 batch 调用 `model(image, geometry, condition)`。 |
| 2 | image encoder | `corejepa/models/encoder.py` | `VisitEncoder3D` 是 4 级 3-D residual CNN（GroupNorm + SiLU），自适应池化后映射到 192 维；随后 `VisitProjector` 用两层 MLP 投到 JEPA state space。输入默认是 8 通道 DCE8。 |
| 3 | EMA target encoder | `corejepa/models/corejepa.py` | online encoder、projector 和 geometry projector 都有不可训练的 deepcopy target；每个优化 step 后以 momentum 0.996 做 EMA。target state 同样是 `target appearance + target geometry state`，并非纯影像 target。 |
| 4 | conditioned causal Transformer | `corejepa/models/transition.py` | `ConditionedCausalTransformer` 加 learned position、`condition_add`，并用 condition 生成 FiLM 的 gamma/beta；采用严格上三角 causal mask。`ImageTransition` 一次输出 T1/T2/T3 三个完整 next-state latent。 |
| 5 | Factorized Response State pathway | `corejepa/models/response_state.py`；`corejepa/models/corejepa.py` | `FutureResponseState` 从 9-D geometry prefix 与 condition 预测 64-D state；6 个 expert 的 gate 读取非时间 condition（治疗与 baseline context）；decoded geometry 与 response state 经 adapter 产生 image prediction 的 latent correction。 |
| 6 | IRG 相关代码 | `corejepa/models/response_state.py`；`corejepa/training/losses.py`；`corejepa/data/response_targets.py` | IRG heads 从 FRS 预测 scalar response score、18-D response vector 及相邻 update，并包含连续排序/对比、route/entropy/balance 项。它约束的是 geometry/condition 驱动的 FRS，不是 predicted image delta，因此不能直接充当本任务 M2 的 radiomics-on-image-change head。 |
| 7 | SIGReg/latent regularization | `corejepa/training/losses.py` | `SIGReg` 对 online `visit_state` 的随机 256 个投影实施各向同性高斯特征函数正则，权重 0.09；另以 `visit_state_std >= 0.05` 作为 checkpoint 合格门槛。 |
| 8 | frozen linear readout | `corejepa/readout/flr.py`；`scripts/fit_readout.py` | class-balanced LogisticRegression，L1/L2 与 C 网格由 validation landmark AUROC 加权选择。正式输入是 `future_response_state`，即 geometry/condition FRS；并不是 image-only。输出 threshold 固定 0.5。 |
| 9 | 五折划分 | `corejepa/data/records.py`；`corejepa/training/runner.py` | clean 原生代码只有一次 pCR-stratified 70/15/15 split（seed 2026），没有五折循环或 fold manifest loader。另找到一个外部候选五折 CSV，风险见第七节。 |
| 10 | T0/T1/T2/T3 组织 | `corejepa/data/contracts.py`；`corejepa/data/dataset.py`；`docs/PIPELINE.md` | 每人 `image [4,8,Z,Y,X]`、`geometry [4,9]`；三个 causal prefix 分别为 `T0→T1`、`T0,T1→T2`、`T0,T1,T2→T3`。context 使用 online visit state，target 使用 EMA 编码的后三访。 |
| 11 | checkpoint 与结果文件 | `corejepa/training/runner.py`；`docs/MODULE_IO.md` | 预期产物为 `best_corejepa.pt`、`last_corejepa.pt`、`history.csv`、`frozen_states.npz`、`splits.json`，FLR 另写 `flr.pkl`、metrics/scores/summary；当前 clean 输出目录不存在，未找到这些产物。 |
| 12 | image-only 配置 | `configs/paper_v1.yaml` 及全 clean 代码搜索 | 未发现 image-only/no-condition/no-geometry 配置。默认固定 `image_channels=8`、`geometry_dim=9`，实例化 condition encoder、FRS 与 IRG。 |
| 13 | clinical、treatment、geometry 进入点 | `corejepa/data/condition.py`；`models/corejepa.py`；`models/transition.py`；`models/response_state.py` | clinical/treatment 形成 25-D condition，进入 image transition 的 additive/FiLM、FRS dynamics 与 expert gate；geometry 投影后直接加进 online/target visit state，同时进入 FRS。 |
| 14 | binary lesion mask 是否为 MRI channel | `corejepa/data/contracts.py`；`corejepa/data/imaging.py` | 是。DCE8 前 7 通道为 pre/early/late、两个差分、peak relative enhancement、washout relative enhancement，第 8 通道 `roi` 是二值 ROI crop。9-D geometry 又由同一 ROI 计算，因此 mask/geometry 信息被双路径暴露。 |
| 15 | next-state loss 精确定义 | `corejepa/models/corejepa.py`；`corejepa/training/losses.py`；`configs/paper_v1.yaml` | target 为 EMA 编码下一访的完整 state；prediction 为 conditioned image prediction 加 FRS correction。主 prediction loss 是逐样本、逐 transition 对预测与 target 分别做 latent 维 LayerNorm 后的 MSE，步权重 2/1/0.5 先归一为均值 1；raw MSE 只记录，不参与优化。总损失还加 SIGReg、IRG score/vector/update、delta contrast 与 routing 项。 |

### 4.2 当前 state、condition 与 next-state objective 的精确定义

当前 online 和 target visit state 都显式含 geometry：

```text
z_online(t) = VisitProjector(VisitEncoder3D(DCE8_t))
              + GeometryProjector(q_t)

z_target(t) = EMAVisitProjector(EMAVisitEncoder3D(DCE8_t))
              + EMAGeometryProjector(q_t)
```

25-D condition 的组成是：

```text
3 个 target-visit one-hot
+ 4 个 observed-prefix mask
+ 14 个 exact treatment-arm one-hot
+ HR、HER2、MammaPrint、age_z
```

预测为：

```text
z_image_hat(t+1) = ImageTransition(z_online(0:t), condition_t)
s_hat(t+1)       = FutureResponseState(q_0:t, condition_t)
z_hat(t+1)       = z_image_hat(t+1) + latent_correction(s_hat, decoded_q)
target(t+1)      = stop_gradient(z_target(t+1))
```

令 `LN` 表示沿 latent 维的 LayerNorm，配置的原始步权为 `(2, 1, 0.5)`，代码除以三项均值后得到 `(12/7, 6/7, 3/7)`。主预测项是：

```text
L_prediction = mean_(patient, step) [
    w_step * mean_latent((LN(z_hat) - LN(target))^2)
]
```

`prediction_raw_mse = mean((z_hat-target)^2)` 仅写入统计。完整 objective 的配置权重为：prediction 1.0、SIGReg 0.09、response score 0.05、state-delta contrast 0.02、update score 0.02、response vector 0.02、response-vector update 0.02、gate route 0.3、gate entropy 0.05、gate balance 0.2。

### 4.3 与 image-only 科学问题直接相关的代码风险

- `ConditionEncoder(records)` 用传入的全部 records 建立 treatment-arm vocabulary，并以全部 records 计算 age 均值/标准差，而不是 fold-train-only；若继续使用 condition，这会造成预处理统计跨 split。新的 image-only 路径应完全移除该输入，或在控制组中改为 fold-train-only 拟合。
- 当前 FLR 虽然没有再拼接 clinical 字段，但其 `future_response_state` 本身由 geometry 和 clinical/treatment condition 生成，所以不能称为 image-only readout。
- 当前 IRG 的 “state delta contrast” 计算的是相邻 FRS state 之差，不是 `z_target(t+1)-z_target(t)` 或 image prediction delta。
- 第 8 个 ROI mask channel 和另一路 9-D geometry 都可能形成 lesion-size/shape shortcut。若保留第 8 通道，实验命名必须是“ROI 辅助 image-only”，不能称为严格无 geometry；严格 image-only 需要另做 7 通道无 mask 检查。

## 五、数据路径与只读文件清单

### 5.1 实际数据根目录

仓库 `$REPO_ROOT/data` 不存在，工作区 `$WORKSPACE_ROOT/data` 也不存在。权威 clean 配置 `ispy_jepa_tmi_clean/configs/paper_v1.yaml` 指向共享路径：

| 路径 | 大小/状态 | 用途 |
|---|---:|---|
| `/data/data/Breast_Cancer/I-SPY2` | 约 2.3 TiB | 原始 I-SPY2 DICOM、clinical Excel、纵向 measurement Excel |
| `/data/data/Preprocessed/I-SPY2` | 约 1.6 TiB | NIfTI/manifest、clinical 索引、派生 measurement 表、候选 cache/fold |
| `/data/data/Preprocessed/I-SPY1` | 约 35 GiB | 156 人额外 pCR-free pretraining 数据源 |

以下清单聚焦递归搜索中与 MRI、patient ID、T0–T3、FTV/LD/sphericity/BPE、clinical/pCR/treatment/biomarker 和 fold 直接相关的文件。共享目录中的数百万 DICOM/NIfTI 文件不逐项复制到报告；MRI 级索引由患者目录下 `manifest.json` 和 `_manifest_audit.csv` 表示。

### 5.2 关键原始与预处理文件

| 路径 | 类型 | 大小 | 结构 | 可能用途/状态 |
|---|---|---:|---|---|
| `/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx` | XLSX | 132,946 B | 1 sheet；384×29 | 原始四访 longitudinal measurement/radiomics；只读权威源 |
| `/data/data/Breast_Cancer/I-SPY2/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` | XLSX | 58,200 B | 1 sheet；985×10 | Patient_ID、Arm、HR、HER2、MP、pCR、Age、人口学字段；只读权威源 |
| `/data/data/Preprocessed/I-SPY2/clinical_labels.csv` | CSV | 290,294 B | 985×26 | 全 clinical 与 MRI 预处理状态索引 |
| `/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv` | CSV | 235,414 B | 808×26 | clean primary cohort、pCR 与 patient ID 索引 |
| `/data/data/Preprocessed/I-SPY2/mri_nact_features_long.csv` | CSV | 278,310 B | 1,536×19 | 384 人×4 访视的 measurement 长表 |
| `/data/data/Preprocessed/I-SPY2/mri_nact_features_wide.csv` | CSV | 163,694 B | 384×38 | measurement 宽表 |
| `/data/data/Preprocessed/I-SPY2/mri_nact_features_complete4visits_wide.csv` | CSV | 159,733 B | 375×38 | 与完整四访 MRI 成功匹配的 measurement 宽表 |
| `/data/data/Preprocessed/I-SPY2/mri_nact_features_with_clinical_labels.csv` | CSV | 198,591 B | 384×48 | measurement+clinical 合并表；仅审计/控制组，不能作为正式 image-only 输入 |
| `/data/data/Preprocessed/I-SPY2/mri_nact_feature_dictionary.json` | JSON | 6,518 B | 字段映射/摘要 | 辅助解释 measurement 字段 |
| `/data/data/Preprocessed/I-SPY2/clinical_label_dictionary.json` | JSON | 3,003 B | 字段映射/摘要 | 辅助解释 clinical 字段 |
| `/data/data/Preprocessed/I-SPY2/_manifest_audit.csv` | CSV | 38,575 B | 985×7 | MRI patient/visit 完整性索引 |
| `/data/data/Preprocessed/I-SPY2/_batch_summary.csv` | CSV | 80,434 B | 985×5 | MRI 预处理 batch 状态与日志索引 |
| `/data/data/Preprocessed/I-SPY2/<patient_id>/manifest.json` | JSON pattern | 每人可变 | T0/T1/T2/T3 的 DCE NIfTI、mask、bbox、phase 等路径 | clean raw-to-tensor 的患者级权威 MRI 索引 |
| `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv` | CSV | 96,384 B | 4,040×4；808 人×5 折 | 候选五折 manifest；结构有效但 provenance 不完整 |
| `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/paired_cache_audit.csv` | CSV | 424,498 B | 邻近 pipeline 的 cache 审计 | 仅作候选五折旁证，不是 clean checkpoint |
| `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/summary.json` | JSON | 984 B | 邻近 pipeline 摘要 | 不能替代 fold 生成 provenance |
| `/data/data/Preprocessed/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96` | directory | 约 68 GiB | 4,820 files，其中 964 个 patient NPZ | 可只读复用的 legacy DCE8 cache；需 wrapper |
| `/data/data/Preprocessed/I-SPY2/_corejepa_clean_dce8` | directory | **不存在** | clean 期望 `image/geometry` NPZ | 直接运行原 trainer 会尝试在共享路径构建，当前不应执行 |
| `/data/data/Preprocessed/I-SPY2/corejepa_response_features.npz` | NPZ | **不存在** | clean 期望 106-D raw response cache | 直接运行原 trainer 会尝试构建，当前不应执行 |

关键只读源的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `Multi-feature-MRI-NACT-Data.xlsx` | `f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc` |
| `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` | `c016962d2d1e23686746ad3e74a58caeb2d1362f6393fd6209c10723f87c3a53` |
| `clinical_labels_complete4visits.csv` | `b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436` |
| 候选五折 manifest | `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` |

## 六、Radiomics/measurement 与 MRI cohort 的实际内容

### 6.1 原始工作簿结构

`Multi-feature-MRI-NACT-Data.xlsx` 只有一个 sheet：`datawith4visits`，384 行、29 列；`CLINICAL-TRIAL-SUBJECT-ID` 唯一，无重复。29 列全部无缺失。字段实际分为：

- patient ID：`CLINICAL-TRIAL-SUBJECT-ID`；
- FTV/肿瘤体积：`VOLUME_TUM_BLU_V10/V20/V30/V40`，分别对应 T0/T1/T2/T3；
- sphericity：`SPHERICITY_T0/T1/T2/T3`；
- longest diameter：`LD_T0/T1/T2/T3`；
- BPE：`BPE_5slice_mean_T0/T1/T2/T3`；
- 从 T0 到 T1/T2/T3 的百分比变化：FTV、Sphericity、LD、BPE 各 3 列，共 12 列。

FTV 的体积单位 `cc` 来自随附 FTV DICOM 文档；LD 和 BPE 的单位未在工作簿/字典中明确给出，不能自行假设。sphericity 为无量纲。源表中的百分比变化列只适合作一致性核验；相邻 T1→T2、T2→T3 change 仍需由原始访视值计算。

原始 clinical 工作簿 sheet 为 `ISPY2_n985_TCIA_clinical`，985×10，字段为 `Patient_ID`、`Arm`、`HR`、`HER2`、`MP`、`pCR`、`Age_at_Screening`、`Race`、`menopausal_status`、`ethnicity`。

### 6.2 ID 匹配与覆盖

使用显式、可复核的六位 ClinicalTrialSubjectID 规则：仅接受 MRI `patient_id` 满足 `^(?:ISPY2-|ACRIN-6698-)(\d{6})$`，提取六位数字后与 `clinical_patient_id` 及工作簿 `CLINICAL-TRIAL-SUBJECT-ID` 等值比较。没有使用 fuzzy matching。

| 项目 | 人数/比例 |
|---|---:|
| 完整四访 MRI primary cohort | 808 |
| 原始 measurement 工作簿 | 384 |
| 可与 808 人严格匹配 | 375（46.41%） |
| 808 人中无 measurement | 433（53.59%） |
| measurement 工作簿中不在 808 cohort | 9 |
| 匹配患者中单项访视缺失 | 0 |

9 个 cohort 外 trial ID 为：`246134`、`495440`、`516763`、`652480`、`745633`、`748611`、`790722`、`889225`、`893874`。它们不能因有 radiomics 而临时加入锁定的 808 人评估 cohort。

### 6.3 Transition 数量

375 名匹配患者的四类特征在四访均完整：

| Transition | MRI cohort 人数 | 有完整 radiomics/measurement 的人数 | 四类特征均有效 |
|---|---:|---:|---:|
| T0→T1 | 808 | 375 | 375 |
| T1→T2 | 808 | 375 | 375 |
| T2→T3 | 808 | 375 | 375 |
| 合计 | — | 1,125 个 patient-transition | 1,125 |

候选五折中，各 split 的 radiomics 患者数为：

| Fold | train | validation | test |
|---:|---:|---:|---:|
| 0 | 247 | 59 | 69 |
| 1 | 239 | 69 | 67 |
| 2 | 240 | 52 | 83 |
| 3 | 242 | 61 | 72 |
| 4 | 225 | 66 | 84 |

因此 M2 的辅助目标可覆盖每折 225–247 名 I-SPY2 train 患者的 675–741 个相邻 transition；其余训练患者仍可参与 image evolution loss，但 radiomics auxiliary loss 必须使用 mask 跳过。

### 6.4 完整病例选择风险

audit 中的粗略完整病例比较显示：radiomics 可用组 pCR 比例为 29.33%（110/375），不可用组为 38.11%（约 165/433）；平均年龄分别为 48.26 与 49.46 岁，亚型与 treatment-family 构成也有差异。不可用组没有同定义的 FTV 字段，无法做 baseline lesion volume 的同源比较。

这说明 375 人并非可默认视为 808 人的随机子样本。M2 不能只在 complete-case 子集训练整个模型；正确边界是所有 MRI train 患者继续贡献 image loss，仅有匹配且目标有效的 transition 贡献 radiomics auxiliary loss，并单独报告 complete-case sensitivity。

## 七、五折、cache、checkpoint 与结果审计

### 7.1 clean 原生 split 与候选五折的差异

clean `stratified_split()` 只生成一次 70/15/15 划分。按当前 808 人 cohort，对应预期规模约为 565 train、121 validation、122 test；I-SPY1 的 156 人只被附加到 pCR-free pretraining train，不进入 validation、test 或 FLR。

候选五折 CSV 的结构检查通过：

| Fold | train | validation | test | 总计 |
|---:|---:|---:|---:|---:|
| 0 | 525 | 121 | 162 | 808 |
| 1 | 525 | 121 | 162 | 808 |
| 2 | 525 | 121 | 162 | 808 |
| 3 | 526 | 121 | 161 | 808 |
| 4 | 526 | 121 | 161 | 808 |

每名患者在每折恰有一行，五折中恰好一次进入 test，`label_pcr` 与患者记录可对齐。它可支撑后续五折实验的结构锁定。

但 provenance 风险必须保留：

- 文件位于 `_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026`，不是 clean 配置目录或当前 branch 的 tracked manifest；
- 当前仓库中搜索不到该文件名、目录名或其生成入口；
- 同目录只有 `paired_cache_audit.csv` 与 `summary.json`，没有完整 fold 生成代码、参数快照或 clean checkpoint；
- 文件修改时间只能说明文件存在，不能证明它就是 clean baseline 当时使用的原生划分；
- clean 原生 trainer 不读取该 CSV。

结论：后续若采用，必须将其视为 hash 锁定的 `valid_candidate_copy`，记录来源和 SHA-256，并对所有 M0/M1/M2 使用同一副本。报告只能称“在候选五折上的新训练比较”，不能称“复现了原 clean 五折数值”。

### 7.2 Cache 兼容性

clean 配置要求的 tensor cache 和 response cache 均缺失。`LongitudinalDCEDataset(build_missing=True)` 会在取样时自动调用 `build_patient_tensor()`，`runner.build_datasets()` 也会在 response cache 缺失时自动构建。因此直接运行原 `pretrain.py` 会写共享数据路径，不符合本实验优先只读复用和避免修改共享 cache 的约束。

legacy cache 的实测特征为：

- 目录约 68 GiB；4,820 个文件，其中 964 个 patient-level NPZ；
- 964 人 = 808 名 I-SPY2 primary + 156 名 I-SPY1 pretraining-only；
- patient NPZ 的影像 key 为 `x`，shape `[4,8,32,96,96]`，float32；
- 第 8 通道是 binary ROI mask；
- clean dataset 则期待以 `<patient_id>.npz` 命名并含 `image` 与 `geometry` key。

因此可通过 additional experiment 中的只读 dataset wrapper 复用 `x`，但不能将 legacy cache 冒充 clean cache，也不能在共享目录改名或补写 key。

### 7.3 Checkpoint/结果缺失

以下预期路径/文件未发现：

- 仓库根或 `ispy_jepa_tmi_clean/` 下的 `runs/corejepa_clean/`；
- `best_corejepa.pt`、`last_corejepa.pt`；
- `history.csv`、`frozen_states.npz`、`splits.json`；
- `flr.pkl`、`flr_metrics.csv`、`flr_scores.csv`、`flr_summary.json`。

所以当前阶段只能复核实现和数据契约，不能报告一个“已复现”的 M0 数字。M0 必须以新建的 image-only/ROI 辅助 wrapper 在统一候选五折上重新训练，并与 M1/M2 同协议比较。

## 八、已生成 data_audit 产物清单

以下文件由独立审计脚本从只读源生成，均位于 `additional_experiments/radiomics_next_change/data_audit/`：

| 文件 | 大小 | 内容与用途 |
|---|---:|---|
| `radiomics_audit_summary.json` | 2,375 B | 原始文件 hash、cohort overlap、transition、fold、cache 和单位摘要 |
| `radiomics_workbook_sheets.csv` | 112 B | Excel sheet 名与维度 |
| `radiomics_schema.csv` | 8,340 B | 29 列 schema、dtype、role、单位、缺失、分位数、异常值与 planned change |
| `radiomics_patient_overlap.csv` | 318,944 B | 808 人逐患者 ID 对齐、radiomics/cache/pCR/clinical/fold 状态 |
| `radiomics_missingness.csv` | 3,515 B | 原始表、808 cohort 与 paired transition 的逐特征缺失率 |
| `radiomics_transition_counts.csv` | 2,020 B | overall 及逐 fold/split/transition 的可用人数 |
| `radiomics_transition_targets_raw.csv` | 208,222 B | 375 人×3 transition 的四项起止值、绝对差和 valid mask |
| `radiomics_complete_case_bias.csv` | 3,808 B | radiomics 可用/不可用组的 pCR、年龄、亚型、MammaPrint、治疗与 fold 构成 |
| `data_file_inventory.csv` | 2,042 B | 关键源文件/目录的存在性、字节数、类型、用途与只读标记 |

这些是数据审计产物，不是 fold-specific 训练变换。FTV/LD 的 epsilon、winsorization、imputation、center/scale 等任何训练统计仍必须在每折 train 患者内重新拟合，不能从全 cohort 审计表反推后用于训练。

## 九、风险登记与后续决策边界

| 风险 | 证据 | 影响 | 必须采取的边界 |
|---|---|---|---|
| baseline 非 image-only | condition、geometry、FRS 和 FLR 的代码路径 | 无法直接作为任务 M0 | 新建无 clinical/treatment/9-D geometry/FRS 的实验 wrapper |
| ROI geometry 双重暴露 | DCE8 第 8 通道 + 9-D geometry projector | copy/shape shortcut，命名易夸大 | 保留 mask 时标为“ROI 辅助 image-only”；另做 7 通道诊断 |
| clean cache 缺失 | 配置路径不存在 | 原 trainer 会写共享路径且重建成本高 | 只读使用 legacy cache wrapper，不触发原自动构建 |
| clean checkpoint 缺失 | 没有 runs/checkpoint/result 文件 | 不能给出 native baseline 数值 | 新训练 M0；明确“实现复核”与“数值复现”的区别 |
| 候选五折 provenance 不完整 | clean branch 无引用/生成器/配套 checkpoint | 不能宣称原生复现 | 锁定 hash，所有模型同 folds，报告限制 |
| radiomics 覆盖仅 46.41% | 375/808；pCR 构成不同 | complete-case selection bias | image loss 用全部 MRI train；辅助 loss masked；报告 complete-case 敏感性 |
| condition 统计跨 split | encoder 以全部 records 算 age/vocab | 若用于控制组则可能泄漏统计 | image-only 正式模型完全移除；控制组 fold-train-only 拟合 |
| 单位不完整 | LD/BPE 单位未明示 | 错误物理解释风险 | 按原始数值和 fold-train transform 处理，不作单位换算 |
| 环境版本漂移与 pytest 缺失 | 实测环境与根 pins 不同 | 复现/测试风险 | 保存版本快照；正式测试时安装或使用等价 unittest/smoke 验证 |

## 十、阶段性判定

没有发现“radiomics 文件缺失、patient ID 无法匹配或时间点无法确认”这类必须暂停的严重阻塞。数据足以继续撰写实验计划并实现 M0/M1/M2，但必须同时满足以下条件：

1. 不把现有 clean 代码称为 image-only，也不把候选五折称为 native clean fold；
2. 不直接启动会写入缺失 shared cache 的原 `pretrain.py`；
3. M0/M1/M2 使用同一 hash 锁定的候选五折和同一 MRI cohort；
4. radiomics 只作为 fold-train 阶段、masked 的 image-change auxiliary target，正式 inference/readout 不读取 radiomics 表；
5. 所有 fold-specific 变换只在 train split 拟合，test 仅在最终锁定评估时读取；
6. 保留第 8 ROI mask 通道时严格使用“ROI 辅助 image-only”措辞，并用无 mask 或其他 shortcut 审计检验结论。
