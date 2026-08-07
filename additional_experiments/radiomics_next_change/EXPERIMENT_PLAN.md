# Radiomics 引导的 Next-Change 实验计划

## 0. 计划状态、研究边界与不可变约束

本计划以仓库当前真实代码、只读数据审计结果和当前可用计算环境为依据。当前 Git 分支为 `feature/ispy-clean-corejepa`，commit 为 `c413ec86af04795434bdc19e65bbb006c966f379`。第一轮必须完成 M0、M1、M2；M3 仅在预先定义的门控条件全部满足后运行。

本研究的主要结论只允许来自“训练时可使用结构化影像测量监督、正式推理时只使用 MRI”的模型。所有主模型均遵守以下约束：

- transition、encoder 和 pCR readout 不读取 clinical profile、treatment arm、9 维 geometry descriptor 或 radiomics 表格；
- pCR 只用于冻结表征后的下游 readout，绝不用于 JEPA/transition/radiomics head 的训练；
- radiomics 只可用于 M2/M3 的训练辅助目标、冻结表征后的 grounding 评估，以及 C0/C1 控制组；
- validation/test 不得拟合任何图像、radiomics 或 readout 预处理统计量；
- test 不得参与模型、超参数、阈值、epoch 或 early stopping 选择；
- 原始数据、原始 manifest、现有 checkpoint 和现有结果均保持只读；新结果只写入本实验目录；
- M0/M1/M2 使用完全相同的 patient fold、随机种子策略、图像输入、训练预算、early stopping 和冻结 readout 协议。

需要特别声明：当前 clean 分支没有可用的 image-only 配置、配套五折 checkpoint 或可直接复算的数值结果，配置所指向的 `_corejepa_clean_dce8` 和 `corejepa_response_features.npz` 也不存在。因此本计划中的运营 M0 是在统一候选五折上**新训练的 ROI 辅助 image-only Next-State 基线**，是对当前架构中图像分支的受控重建，不是已有 native checkpoint 的数值复现。最终报告不得使用“成功复现原论文/原分支数值”之类表述。

## 1. 研究假设

### H1：Next-Change 减少复制当前状态

相邻 MRI 的静态解剖与患者身份信息高度相似，直接预测完整下一状态时，`z_hat_(t+1) ≈ z_t` 可能已获得较低损失。若 transition 显式预测 EMA latent 坐标系中的变化，并同时约束变化与重建状态，则 M1 相比 M0 应表现为：

- 更低的 learned-next-state error；
- 更高且更多为正的 normalized transition gain；
- predicted delta norm 不再塌缩到接近零；
- predicted/target delta cosine similarity 提高；
- repeated-T0 和 temporal shuffle 后的预测与 native 输入产生可解释的差异。

### H2：privileged radiomics supervision 改善 image-change 表征

若 M2 的 radiomics head 只读取 predicted image delta，且 radiomics 不进入正式推理，那么 image delta 对 FTV、LD、sphericity 和 BPE 变化的可预测性应提高；冻结的 image-only pCR readout 也可能优于 M1，尤其在随访 MRI 已观察到的 `T0–T1` 和 `T0–T2` decision point。

### H3：获益可能在较少训练数据时更明显

结构化影像测量提供低维、临床可解释的训练信号，因此若主五折结果稳定，M2 相对 M0/M1 的增益可能在 25% 或 50% 训练患者时更明显。该假设属于主五折完成后的次级分析，不得牺牲 M0/M1/M2 完整五折。

### H4：radiomics complete-case 选择偏差会限制外推

808 名 MRI 完整四访视患者中只有 375 名可匹配 radiomics。审计中 radiomics 可用组 pCR 比例为 29.3%，不可用组为 38.1%，因此只在 complete-case 子集训练或报告很可能产生选择偏差。M2 必须让全部 MRI 训练患者参与 image evolution loss，并用 mask 仅对可匹配 transition 启用 radiomics loss。

## 2. 当前 Next-State 方法的真实实现

当前入口为 `ispy_jepa_tmi_clean/scripts/pretrain.py`，实际训练逻辑位于 `corejepa/training/runner.py`。关键实现如下：

1. `corejepa/models/encoder.py` 的 `VisitEncoder3D` 和 `VisitProjector` 生成 appearance state；`GeometryProjector` 将 9 维 lesion geometry 投影后与 appearance 相加。
2. `corejepa/models/corejepa.py` 同时维护 online encoder/projector 和 EMA target encoder/projector；当前 target state 也加入 target geometry projection，并非纯图像状态。
3. `corejepa/models/transition.py` 的 `ConditionedCausalTransformer` 使用 learned position、condition additive projection 和 FiLM。condition 包含 temporal/prefix、treatment、HR、HER2、MammaPrint、age 等信息。
4. image transition 直接输出下一完整 latent state，而不是 latent change。
5. `corejepa/models/response_state.py` 的 Factorized Response State（FRS）读取 geometry 与 condition，经 expert routing 生成 response latent correction；最终 prediction 是 image prediction 与该 correction 之和。
6. 当前 prediction loss 先对 prediction/target 分别做 LayerNorm，再计算 MSE，三个 transition 的权重来自 `[2.0, 1.0, 0.5]`；raw MSE 仅作诊断。同时存在权重 0.09 的 SIGReg，以及 response guidance、update、contrast 和 routing 等损失。
7. `corejepa/readout/flr.py` 当前读取的是 geometry/condition 驱动的 `future_response_state`，不是 image transition latent，因此当前 FLR 也不能作为本研究所需的 image-only readout。
8. 当前 clean 配置只生成一次 70/15/15 split，没有本研究需要的原生五折产物。

因此，M0 不是直接运行现有 `paper_v1.yaml`。实验代码将复用 `VisitEncoder3D`、`VisitProjector`、EMA 更新、causal Transformer 的主体结构与必要的 SIGReg，但移除 GeometryProjector、condition projection/FiLM、FRS、IRG/guidance/routing 路径，并实现新的 image-derived frozen readout。

## 3. Copy-current shortcut 风险与基线定义

训练和评估时，对每个相邻 transition 定义：

```text
z_copy_(t+1) = stopgrad(z_target_t)
E_copy = d(z_copy_(t+1), z_target_(t+1))
E_model = d(z_hat_(t+1), z_target_(t+1))
G = (E_copy - E_model) / (E_copy + 1e-8)
```

其中 `d` 同时报告 raw MSE 和与训练主目标一致的 normalized latent MSE。`G > 0` 表示 learned transition 优于复制当前状态，`G = 0` 表示无改善，`G < 0` 表示不如复制。由于当 target delta 极小时分母可能放大噪声，还需同时报告 `E_copy`、`E_model`、每个 transition 的分布、median/IQR，以及按 `E_copy` 分位数分层的结果，不能只看平均 `G`。

当前可用 legacy DCE8 cache 的每个患者文件包含 key `x`，形状为 `[4, 8, 32, 96, 96]`。前 7 个通道是 DCE intensity/kinetic 图像通道，第 8 个通道是 binary ROI mask。主五折为与当前 DCE8 设置可比，M0/M1/M2 均保留第 8 通道，因此统一命名为“**ROI 辅助 image-only**”，不能简称为“纯 image-only”。它不读取独立 9 维 geometry descriptor，但 mask 本身仍携带 lesion location/shape shortcut。若主实验资源允许，将在不重建 cache 的前提下于加载时丢弃第 8 通道，运行 7 通道 mask-removed 敏感性分析；由于原 crop 仍由 ROI 流程产生，该分析也不能被解释为完全消除 ROI 先验。

## 4. 真实数据字段、患者匹配与五折来源

### 4.1 Radiomics/measurement 表格

真实源文件为：

`/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx`

工作表 `datawith4visits` 有 384 行、29 列、384 个唯一患者，无重复 patient ID，核心字段无缺失。该文件是宽表结构，不存在独立 `timepoint`、`feature_name` 或 `measurement_value` 列；时间点与特征由列名确定性解析。实际字段为：

- patient ID：`CLINICAL-TRIAL-SUBJECT-ID`，六位整数；
- FTV：`VOLUME_TUM_BLU_V10`、`V20`、`V30`、`V40`，对应 T0、T1、T2、T3；FTV 单位 cc 来自随附 DICOM 说明；
- sphericity：`SPHERICITY_T0` 至 `SPHERICITY_T3`，无量纲；
- longest diameter：`LD_T0` 至 `LD_T3`；源工作簿/字典未明确单位，不作单位换算或假设；
- BPE：`BPE_5slice_mean_T0` 至 `BPE_5slice_mean_T3`；源工作簿/字典未明确单位，按原始数值处理；
- 12 个源表 baseline-relative percent-change 字段：FTV、Sphericity、LD、BPE 各自的 `T0_T1`、`T0_T2`、`T0_T3` 百分比变化。它们只用于一致性核验与次级分析，不作为第一轮 M2 的主目标。

第一轮 `r_t` 固定为 `[FTV_t, sphericity_t, LD_t, BPE_t]`。不得把该 4 维 measurement 集合误称为已经审计过的高维 radiomic feature bank。

### 4.2 显式 ID 匹配

808 名 MRI cohort 的 patient ID 格式为 `ISPY2-######` 或 `ACRIN-6698-######`。只接受正则：

```text
^(?:ISPY2-|ACRIN-6698-)(\d{6})$
```

提取的六位 ClinicalTrialSubjectID 必须与 clinical 表中的 `clinical_patient_id` 及工作簿 ID 完全相等。匹配前显式检查大小写、首尾空格、连字符、六位前导零和重复 ID；不进行 fuzzy matching。审计得到：

- MRI 完整四访视 cohort：808 人；
- radiomics 工作簿：384 人；
- 精确匹配：375 人，占 MRI cohort 的 46.41%；
- MRI cohort 中无 radiomics：433 人；
- 工作簿中另有 9 人不属于当前 808 cohort；
- 375 名匹配患者四个特征的 T0–T3 均完整，因此 T0→T1、T1→T2、T2→T3 各有 375 个有效 transition，共 1125 个相邻 paired transition。

任何实现若得到不同数量，必须在正式训练前停止并重新审计，不允许为凑数放宽匹配规则。

### 4.3 候选五折 manifest 及 provenance 限制

当前找到的五折文件为：

`/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv`

其 SHA256 为 `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。文件有 4040 行、808 个唯一患者、5 个 fold；每个患者在每折出现一次且恰好一次进入 test。fold 0–2 为 525/121/162（train/val/test），fold 3–4 为 526/121/161。配对患者数如下：

| fold | train paired | val paired | test paired |
|---:|---:|---:|---:|
| 0 | 247 | 59 | 69 |
| 1 | 239 | 69 | 67 |
| 2 | 240 | 52 | 83 |
| 3 | 242 | 61 | 72 |
| 4 | 225 | 66 | 84 |

该文件通过 patient-level disjoint、每名患者 test 一次、label 与 808 cohort 一致等内部检查，但位于另一 BreastDCEDL 产物目录，clean 分支没有引用它，也没有配套 checkpoint 或生成日志。因此它只能作为**锁定的、内部一致的候选五折副本**使用，provenance 状态必须随所有结果记录为 `valid_candidate_copy`。不得宣称这是当前 clean 分支的原生五折，也不得据此宣称 native numerical reproduction。若后续找到权威原始五折，应先比较 SHA、patient assignment 和生成逻辑；不得在看过 test 结果后替换 manifest。

### 4.4 MRI cohort 与 I-SPY1 的处理

主评估 cohort 固定为上述 808 名 I-SPY2 患者。legacy cache 还含 156 名 I-SPY1 完整访视患者，与当前 clean 配置的“外部训练患者”设计一致。为尽量保持现有训练数据策略，这 156 人可在每个 fold 中仅参与 image evolution/SIGReg loss，并在 M0/M1/M2 间完全一致；他们没有 radiomics loss，不进入 validation、test、readout、OOF 统计或超参数选择。训练日志必须分别记录 I-SPY2 与 I-SPY1 样本数。若实现阶段不能证明 I-SPY1 的 ID 无重叠或 visit 语义一致，则主分析改为仅用 I-SPY2，并把该决定写入运行配置与最终报告，而不是静默混入。

## 5. Fold 内 Radiomics Change Target

### 5.1 主目标定义

第一轮只使用相邻变化：

```text
FTV: delta = log(FTV_(t+1) + epsilon_ftv) - log(FTV_t + epsilon_ftv)
LD:  delta = log(LD_(t+1)  + epsilon_ld)  - log(LD_t  + epsilon_ld)
sphericity: delta = sphericity_(t+1) - sphericity_t
BPE:         delta = BPE_(t+1) - BPE_t
```

LD 可为 0，因此 `epsilon_ld` 必须为正。每个 fold、每个需要 log 的特征，仅用该 fold 的 I-SPY2 **training patients** 拟合 `epsilon = max(1e-6, 0.5 × 最小正训练值)`。若训练集中没有正值则该特征在该 fold 标为不可用并停查，而不是从 val/test 获取 epsilon。

计算 raw delta 后，对每个 feature 分别用 training paired transitions 拟合：

1. 1% 和 99% winsorization 边界；
2. winsor 后的 median；
3. `scale = max(IQR, 1e-6)`；
4. 最终 target 为 `(clip(delta) - median) / scale`。

validation/test 只应用相应 fold 的训练参数。每折参数保存到 `configs/radiomics_transform_fold_{k}.json`，同时记录训练 patient ID 的哈希、paired transition 数、feature 顺序、epsilon、clip、median、IQR 和版本。不得先在 375 人全体上标准化再切 fold。

源表的 T0-relative percent change 将用于核对方向与计算次级 `delta_r_(0→t)`；第一轮 M2 不同时加入相邻和 baseline-relative 两套监督，以避免在样本量有限时重复加权同一信号。

### 5.2 缺失与 mask

构造 `target [patient, transition, 4]` 和同形状 `valid_mask`。某个原始端点缺失、非有限或 fold 训练参数不可用时，只把对应 feature mask 置 0；不填造 target，不删除患者、transition 或 batch。433 名无 radiomics 的 MRI 患者 radiomics mask 全为 0，但仍计算完整 image loss。Masked SmoothL1 定义为：

```text
L_rad = sum(mask * smooth_l1(pred, target)) / max(sum(mask), 1)
```

mask 总和为 0 的 batch 将 `L_rad` 置为可微的 0。不得过采样 paired 患者；每个 batch 记录 paired patients、valid feature cells、paired/unpaired 比例以及未加权/加权 `L_rad`。

## 6. M0、M1、M2、M3 模型定义

### 6.1 共同图像骨干

共同骨干复用当前参数规模：8 个输入通道、base channels 16、latent dim 192、predictor depth 3、heads 4、MLP dim 512、dropout 0.1。主模型只包含：

- online `VisitEncoder3D + VisitProjector`；
- EMA target `VisitEncoder3D + VisitProjector`，momentum 0.996；
- 只含 learned position 与 causal self-attention 的 unconditioned Transformer；
- 必要的 prediction head；
- SIGReg（权重 0.09）以控制表征塌缩。

图像 batch 接口不得出现 clinical、treatment、geometry 或 pCR tensor。dataset 可以携带 `patient_id`、fold、visit availability 和 radiomics mask 用于索引/损失路由，但模型 forward 不接收这些受禁输入。

### 6.2 M0：ROI 辅助 image-only Next-State

```text
z_online_0:t = E_online(x_0:t)
z_target_t+1 = stopgrad(E_ema(x_t+1))
z_hat_t+1 = T_state(z_online_0:t)
L_state = weighted normalized-MSE(z_hat_t+1, z_target_t+1)
L_M0 = L_state + 0.09 * L_SIGReg
```

保留当前 `[2.0, 1.0, 0.5]` 的 transition step 权重，并额外记录 raw MSE。M0 不实例化 GeometryProjector、condition layers、FRS、response correction、IRG/guidance 或 routing loss。为与 M1/M2 使用统一诊断，定义其 predicted delta 为 `delta_z_hat = z_hat_(t+1) - stopgrad(z_target_t)`。

M0 是新训练的受控基线。没有 native image-only checkpoint 可供数值复现；只有当训练日志、checkpoint、配置与 OOF prediction 均完整时，才称为“运营 M0 完成”。

### 6.3 M1：ROI 辅助 image-only Next-Change

第一版把 delta target 完全定义在同一个 EMA target 坐标系中：

```text
delta_z_target = stopgrad(z_target_(t+1) - z_target_t)
delta_z_hat = T_change(z_online_0:t)
z_hat_(t+1) = stopgrad(z_target_t) + alpha_delta * delta_z_hat
```

选择 EMA-current 而不是 online-current 构造 target 的原因是避免 online/EMA 坐标漂移混入真实纵向变化，并阻断 target 分支梯度。transition 上下文仍来自 online encoder；EMA current 仅作为训练重建锚点，正式推理时保存的 EMA image encoder 同样可以从已观察当前 MRI 生成该锚点，因此没有使用真实未来信息。

第一轮锁定 `alpha_delta = 1.0`，不使用 test 调整。至少比较：

- M1-delta：`L = SmoothL1(delta_z_hat, delta_z_target) + 0.09 * SIGReg`；
- M1-delta+state：在上述损失上加入与 M0 相同的 normalized state reconstruction loss。

两项均使用相同 step 权重。默认 `lambda_delta = 1`、`lambda_state ∈ {0, 1}`；先在 fold 0 仅依据 validation delta cosine、state error 和 transition gain 做一次受控选择，选择规则和结果在查看任何 test 指标前锁定，再用于五折。若两者相当，预先选择 `delta+state`，因为它直接保持与 M0 的 next-state 可比性。训练必须记录 predicted/target delta norm、delta cosine、raw/normalized next-state error、copy error 和 normalized transition gain。

fold 0 正式 validation pilot 已在任何 M1 test 读取前完成。第 12 epoch 的 delta-only/raw gain、native normalized gain、delta cosine、normalized state loss 分别为 `+0.108/-0.014/0.256/0.615`；delta+state 分别为 `-0.041/+0.036/0.076/0.586`。两者揭示 raw latent 与原 clean LayerNorm 距离的坐标权衡。由于 delta+state 是唯一在原生 normalized distance 上优于 copy 且 state error 更低的版本，按预注册 tie-break 锁定 **delta+state** 作为正式 M1/M2 基础；raw gain 与 cosine 的代价必须在最终报告中如实呈现，不能只报有利坐标。

### 6.4 M2：Next-Change + Delta-Radiomics 辅助监督

M2 继承锁定后的 M1。轻量 MLP head 只读取 predicted image delta：

```text
delta_r_hat = H_rad(delta_z_hat)
L_M2 = L_M1 + lambda_rad * MaskedSmoothL1(delta_r_hat, delta_r_target)
```

head 的输出顺序固定为 `[FTV, sphericity, LD, BPE]`。不得读取 `z_target_(t+1)`、真实未来 MRI、radiomics input、clinical、treatment 或 pCR。所有 MRI 患者继续贡献 `L_M1`；仅 mask 为 1 的 feature cell 贡献 `L_rad`。

候选 `lambda_rad = {0.05, 0.1, 0.25, 0.5}`。在 fold 0 的 validation 上先排除造成 image validation loss 比无辅助 M1 恶化超过 5%、梯度异常或 delta collapse 的候选，再在剩余候选中选择 validation radiomics standardized MAE 最低者；若差异小于 1%，取较小 lambda。选择不使用 pCR/test，锁定后运行五折。记录 raw `L_rad`、加权项、各 loss 的梯度范数和 paired/unpaired 比例，避免辅助项仅因数值尺度主导训练。

### 6.5 M3：可选 relational loss

M3 只在 M0/M1/M2 五折和 grounding/shortcut 检查完成后考虑：

```text
D_img(i,j) = pairwise distance(delta_z_hat_i, delta_z_hat_j)
D_rad(i,j) = pairwise distance(delta_r_i, delta_r_j) over joint valid features
L_rel = |normalize(D_img) - normalize(D_rad)|
L_M3 = L_M2 + lambda_rel * L_rel
```

pair 仅从 training fold 构造，优先限制在相同 HR/HER2 subtype、相同/相近 treatment family 和相近 baseline burden 的患者内。这些变量只用于训练 pair 的分层索引，不输入 encoder/transition/readout。不得使用 pCR 构造 pair。共同有效 radiomics feature 太少或合格 pair 少于预设最小数时跳过该 batch 的 `L_rel`。

运行 M3 必须同时满足：M2 无 NaN/Inf 且梯度稳定；validation `L_rad` 优于 training-fold mean predictor；至少一个 radiomics feature 在 validation OOF-style 预测中显示非零可预测性；M2 image-only 平均 AUROC 相对 M1 未下降超过 0.02；没有发现 ID、时间点、fold 或 transform 泄漏。任一条件不满足，M3 状态记录为“按门控未运行”，不是缺失结果。

## 7. 计划新增的代码与配置

所有实现限定在 `additional_experiments/radiomics_next_change/`，主 clean 代码保持不变。计划文件如下：

| 文件 | 作用 | M0 | M1 | M2/M3 |
|---|---|:---:|:---:|:---:|
| `src/rnc/data.py` | 读取只读 legacy NPZ、候选五折、严格 patient ID；生成 image prefix 与 radiomics mask | ✓ | ✓ | ✓ |
| `src/rnc/model.py` | image-only online/EMA encoder、无条件 causal transition | ✓ | ✓ | ✓ |
| `src/rnc/losses.py` | state、delta、SIGReg、masked radiomics 与 copy 指标 | ✓ | ✓ | ✓ |
| `src/rnc/transforms.py` | fold-train-only transform 的 fit/apply/save 与时间点对齐 |  |  | ✓ |
| `src/rnc/evaluation.py` | frozen readout、OOF 指标、grounding、copy gain 与 shortcut | ✓ | ✓ | ✓ |
| `scripts/train_model.py` | 单模型单 fold 训练、smoke、checkpoint/history | ✓ | ✓ | ✓ |
| `scripts/fit_radiomics_transforms.py` | 生成五个 fold 的 transform JSON |  |  | ✓ |
| `scripts/evaluate_fold.py` | 单 fold 冻结提取、validation-only readout、prediction-level CSV、M2 原生 grounding 与 repeated-T0/shuffle/copy | ✓ | ✓ | ✓ |
| `scripts/select_m2_lambda.py` | fold 0 validation-only λ 门控；frozen 指标重算、transform refit 与 provenance 审计 |  |  | ✓ |
| `src/rnc/controls.py` | C0/C1/C2 与统一 post-hoc Ridge grounding |  |  | ✓ |
| `scripts/run_controls.py` | C0/C1/C2 和 radiomics subset 比较 |  |  | ✓ |
| `src/rnc/aggregation.py` | 严格五折完整性检查、patient bootstrap、表格与作图 | ✓ | ✓ | ✓ |
| `scripts/aggregate_results.py` | 汇总 OOF、五折均值/标准差、图表与中文结果表 | ✓ | ✓ | ✓ |
| `configs/base.yaml` | 共同数据、骨干、seed、预算和输出约束 | ✓ | ✓ | ✓ |
| `configs/m0.yaml`、`m1.yaml`、`m2.yaml` | 仅声明各模型差异，避免隐式参数漂移 | ✓ | ✓ | ✓ |

实际 checkpoint 路径固定为 `checkpoints/<run_name>/fold_<k>/best.pt`；seed、resolved config/source/data hashes 保存在安全 checkpoint payload 与 `resolved_run.json`，evaluation namespace 再加入 checkpoint/evaluator hash。所有目录用原子 claim/create-new 写入并默认拒绝覆盖。

## 8. 训练顺序与命令

以下是实现后固定使用的命令接口；所有命令从仓库根目录执行，并使用 conda 环境 `bowen`。正式训练前先保存环境、Git、manifest/cache SHA 和 audit summary 到每个 run 的 provenance 文件。

### 8.1 数据与 transform

```bash
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/audit_radiomics.py
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/fit_radiomics_transforms.py
```

### 8.2 Smoke test

```bash
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m0.yaml --run-name m0 --fold 0 --epochs 2 --smoke-patients 24
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m1_delta_only.yaml --run-name m1_delta_only --fold 0 --epochs 2 --smoke-patients 24
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m1.yaml --run-name m1 --fold 0 --epochs 2 --smoke-patients 24
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m2.yaml --run-name m2 --fold 0 --epochs 2 --smoke-patients 24 --lambda-rad 0.1
```

Smoke 使用少量 training patients 和 1–3 epoch，但仍使用真实 fold 0 training transform；不得用 validation/test target 拟合任何参数。

### 8.3 正式五折

M0 完成后才启动 M1；M1 选定 variant 后才启动 M2。单 fold 命令为：

```bash
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m0.yaml --run-name m0_final --fold 0
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m1.yaml --run-name m1_final --fold 0
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/train_model.py --config additional_experiments/radiomics_next_change/configs/m2.yaml --run-name m2_final --fold 0 --lambda-rad 0.05
```

实际将 `--fold` 替换为 0–4。正式 validation pilot 已锁定 M1=`delta+state`、M2 `lambda_rad=0.05`；v3 selector 报告位于 `reports/m2_lambda_selection_v3.md`。最多使用三张 GPU 各运行一个独立 fold；不得让不同作业写同一输出目录。

### 8.4 Readout、grounding、shortcut 和控制组

单个模型/单 fold 的 `evaluate_fold.py` 一次完成冻结表征、train-only logistic、validation-only 超参与阈值、native prediction、copy、repeated-T0、temporal-shuffle 和 M2 原生 head grounding：

```bash
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/evaluate_fold.py \
  --checkpoint additional_experiments/radiomics_next_change/checkpoints/m2_final/fold_0/best.pt \
  --config additional_experiments/radiomics_next_change/configs/m2.yaml \
  --device cuda --batch-size 16 --workers 4
```

将 checkpoint/config 分别替换为 M0、M1、M2 和 fold 0–4。控制组也是单 fold 命令：

```bash
conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/run_controls.py \
  --m0-checkpoint additional_experiments/radiomics_next_change/checkpoints/m0_final/fold_0/best.pt --m0-config additional_experiments/radiomics_next_change/configs/m0.yaml \
  --m1-checkpoint additional_experiments/radiomics_next_change/checkpoints/m1_final/fold_0/best.pt --m1-config additional_experiments/radiomics_next_change/configs/m1.yaml \
  --m2-checkpoint additional_experiments/radiomics_next_change/checkpoints/m2_final/fold_0/best.pt --m2-config additional_experiments/radiomics_next_change/configs/m2.yaml \
  --output-name c0_c1_c2 --device cuda --batch-size 16 --workers 4

conda run --no-capture-output -n bowen python additional_experiments/radiomics_next_change/scripts/aggregate_results.py \
  --output-tag final_analysis_v2 --controls-name c0_c1_c2
```

上述正式入口均已通过 smoke 并完成五折；输出目录默认拒绝覆盖。

## 9. 训练预算、随机性与预计资源

共同训练预算继承 clean 配置：batch size 32、最多 12 epoch、patience 4、learning rate `5e-5`、weight decay `1e-4`、EMA momentum 0.996。每折使用 `seed = 2026 + fold`，同一 fold 的 M0/M1/M2 共用 seed、dataloader 顺序和 patient sampling；所有 seed、确定性选项和实际停止 epoch 写入日志。

当前机器有 3 张 NVIDIA RTX PRO 6000 Blackwell Max-Q，每张约 97.9 GB 显存。legacy cache 已存在且约 68 GB，因此不计划重建共享图像 cache。初始资源预算为：

| 阶段 | GPU | 预计墙钟时间 | 说明 |
|---|---:|---:|---|
| 数据审计与 transforms | CPU | 5–20 分钟 | 只读 Excel/CSV 与小型 JSON/CSV 输出 |
| 单模型 smoke | 1 | 5–20 分钟 | 1–3 epoch，小患者子集 |
| 单模型单 fold 正式训练 | 1 | 10–30 分钟 | 以第一次完整 M0 实测为准 |
| M0/M1/M2 共 15 个主 fold | 最多 3 | 约 1–4 小时墙钟、约 5–10 GPU 小时 | 不含 pilot grid；三路并行 |
| M1 variant 与 M2 lambda pilot | 1–3 | 约 1–3 小时 | 仅 fold 0 validation，不读取 test |
| readout/grounding/shortcut/聚合 | CPU 或 1 GPU | 0.5–2 小时 | 取决于是否重跑 encoder 提取 |

以上是规划值，不是已测性能。M0 首个完整 fold 后更新实测吞吐、峰值显存和剩余预算。若单 fold 明显超过估计，不减少五折或更改模型间共同预算；优先停止可选 M3、7 通道敏感性和数据效率实验。

## 10. Frozen image-only pCR readout

训练完成后冻结 online/EMA encoder 和 transition。每个 decision point 只允许使用已观察 MRI：

- T0：观察 `x_0`，预测 T1 latent；
- T0–T1：观察 `x_0,x_1`，预测 T2 latent；
- T0–T2：观察 `x_0,x_1,x_2`，预测 T3 latent。

不得用相应的真实未来 MRI 或 radiomics。为保证 M0/M1/M2 feature 维度完全一致，对 decision point `t` 构造：

```text
phi_t = concat(
    z_target_t,
    z_hat_(t+1),
    delta_z_hat_t
)
```

M0 的 `delta_z_hat_t = z_hat_(t+1) - z_target_t`；M1/M2 使用 transition 直接预测的 delta。所有分量均由 MRI 和保存的图像模型产生。每个 decision point 分别拟合相同规格的 class-balanced logistic regression，沿用原项目 `l1/l2` 与 `C ∈ {0.001,0.003,0.01,0.03,0.1,0.3,1,3,10}` 网格；StandardScaler 和模型仅在该 fold train 拟合，validation 选择 penalty/C，并由 validation Youden 指数选择阈值。不得在 test 调整任何 readout 参数；同时另报固定 0.5 阈值作为敏感性结果。

报告每 fold 的 AUROC、AUPRC、accuracy、sensitivity、specificity，五折 mean ± std，以及每名患者恰好一次 test prediction 组成的 pooled OOF AUROC/AUPRC。置信区间采用 patient-level bootstrap。冻结 readout 不反向更新 image model。

prediction-level CSV 至少包含：`patient_id, fold, split, model_name, decision_point, y_true, predicted_probability, predicted_label, threshold, checkpoint, has_radiomics, available_visits`。还需保存 manifest、checkpoint、config 和 feature schema 的哈希。

## 11. Transition 与 Radiomics Grounding 指标

### 11.1 Transition 指标

M0/M1/M2 按 fold、split、T0→T1/T1→T2/T2→T3 报告：

- raw 与 normalized next-state MSE；
- copy-current error；
- normalized transition gain 的 mean、median、IQR 与正值比例；
- predicted delta norm、target delta norm 与其比值；
- delta cosine similarity；
- latent 每维标准差、SIGReg、梯度范数和 NaN/Inf 数；
- 按 target-delta/copy-error 分位数分层的结果。

### 11.2 Radiomics grounding

M2 原生 head 在配对 OOF 患者上按 feature、transition、fold 报告 inverse-transform 后的 MAE、RMSE、Spearman、适用时 Pearson、R² 和变化方向准确率；同时报告 standardized 空间指标。prediction CSV 至少包含：`patient_id, fold, transition, feature_name, target_change, predicted_change, valid_mask`。

为公平比较 M0/M1/M2 表征，另在每个 fold 的 frozen `delta_z_hat` 上拟合同一规格的 training-only 线性 radiomics probe，并在该 fold val/test 评估；该 probe 不参与模型训练或 pCR readout。原生 M2 head 与统一 post-hoc probe 分开呈现，不能混为同一结果。

每个 feature 还需画真实/预测分布与散点，并和 training-fold mean-change predictor 比较；报告预测方差/目标方差、接近常数预测比例、方向类别不平衡和样本数，防止低 MAE 掩盖“只预测平均变化”。

## 12. Shortcut resistance 评估

优先复用现有未跟踪的 `shortcut_audit/` 中 fold、perturbation、copy gain 和 provenance 工具，通过本实验 adapter 读取新 prediction；不覆盖其中任何现有文件。

1. **Copy-current**：比较 learned next state 与 `z_target_t`，报告第 11.1 节全部指标。
2. **Repeated-T0**：T0–T1 输入替换为 `[T0,T0]`，T0–T2 替换为 `[T0,T0,T0]`；learned time/position embedding 保持原位置，mask 与图像一起来自重复 T0。模型和 readout 不重训。
3. **Temporal shuffle**：T0–T2 的 `[T0,T1,T2]` 改为 `[T0,T2,T1]`，MRI 所有 image-derived channel 一起交换，position embedding 不交换；不引入 radiomics 表格。T0 和仅两个访视的 T0–T1 不报告不存在的三时点 shuffle。

按模型、fold、decision point 报告 native 与 perturbation 的 AUROC/AUPRC、`ΔAUROC`、`ΔAUPRC`、patient-level probability absolute change、方向变化，以及 transition gain。使用相同 OOF 患者做 paired bootstrap。解释时不把“扰动下降越大”单独视作更好：必须联合 native 性能、future-change 误差、copy gain 和扰动后 patient probability 变化判断模型是否真正使用 follow-up。

额外敏感性包括加载时去除 ROI mask 的 7 通道实验，以及在资源允许时的 mask-only probe，用于量化 binary mask shortcut；它们不替代 repeated-T0 与 temporal shuffle 主诊断。

## 13. 必须的控制组与数据效率

### C0：Radiomics-only logistic regression

仅在 radiomics 可用 subset 上评估。T0 只使用可观察 `r_0`；T0–T1 可使用 `r_0,r_1,delta_r_01`；T0–T2 可使用截至 T2 的值与已观察变化。所有 transform 与 logistic preprocessing 均由 fold train 拟合，不读取未来 timepoint。该控制量化 radiomics 本身的 pCR 预测能力，不支持 image-only 主张。

### C1：Direct Fusion Control

在相同 radiomics subset 上拼接冻结 `phi_t` 与 C0 的可观察 radiomics feature，使用相同 logistic protocol。它在正式使用时依赖 radiomics，仅用于估计“表格在推理时直接可见”的上界，不作为主要方法。

### C2：M2 Image-only Inference

使用 M2 训练好的 frozen `phi_t`，readout 完全不含 radiomics、clinical、treatment 或 geometry。C2 同时在完整 808 OOF cohort 和 radiomics subset 上报告；完整 cohort 是主要结果。C0/C1/C2 的 subset 比较必须使用同一批患者，避免样本构成造成假增益。

### 数据效率（主五折完成后的可选项）

固定各 fold 的 validation/test，不变更 transform 规则；在 train 内以 pCR、HR/HER2 subtype 和 radiomics paired 状态联合分层抽取 25%、50%、75%、100%，比较 M0/M1/M2。若资源不足，只运行 25%、50%、100%。子集 patient IDs 在任何训练前固定并保存；每个比例下 M0/M1/M2 使用相同患者。该分析只有在主五折、shortcut 和控制组均完整后运行。

## 14. 数据泄漏防护与审计断言

每个 run 启动前必须通过以下自动断言：

- 当前分支和 commit 已记录，工作树中的既有用户修改未被清除；
- manifest SHA 与锁定值一致，每 fold train/val/test patient-level disjoint，每人 test 恰好一次；
- legacy cache patient ID 与 cohort 一一对应，NPZ key/shape/visit 顺序为预期；
- radiomics 只通过严格六位 ID 映射，375/433/9 数量与审计一致，无 fuzzy match；
- radiomics transform 的 fit patient 集是当前 fold train paired 集的子集，且与 val/test 交集为空；
- image model forward signature 不含 clinical、treatment、geometry、radiomics 或 pCR；
- M2 head 输入张量只来自 `delta_z_hat`，不来自 target/future truth；
- readout feature schema只包含第 10 节 image-derived 项；
- pCR 仅在 readout 阶段加载；
- checkpoint selection 和 early stopping 不读取 pCR test 或 radiomics test；
- prediction-level CSV 行数、唯一 patient/fold/decision point 和 checkpoint provenance 完整。

clinical/treatment 可以在 complete-case 描述性审计、M3 training-pair 分层和数据效率分层中作为索引元数据使用，但不得成为 M0/M1/M2 tensor 输入。complete-case 比较只作描述性报告，不根据 test outcome 调整训练策略。当前 radiomics 不可用组没有同定义 baseline FTV，故 baseline lesion volume 的两组比较标记为“不可同定义比较”，不能用另一来源变量冒充。

## 15. Smoke 验证、失败标准与停止条件

### 15.1 Smoke 必检项

- 单 batch 的 patient、T0–T3、transition 与 radiomics target 人工抽样核对；
- forward/backward、AMP（如启用）、optimizer、EMA update 和 checkpoint save/reload；
- output/target/mask shape，M2 的 mask=0 患者仍有 image gradient；
- 没有 NaN/Inf，loss scale 和 gradient norm 有限；
- `lambda_rad=0` 的 M2 在相同 seed 下与 M1 loss/gradient 一致到数值容差；
- delta 为零的构造样本、identical-visit copy baseline、feature-level partial mask 的单元测试；
- validation pipeline 不 fit transform，checkpoint 选择可重现；
- readout 和 perturbation 管道无法访问真实未来 MRI/radiomics。

Smoke 结果写入 `reports/smoke_test_report.md` 后才进行正式训练。

### 15.2 立即停止该 run 的条件

- ID/fold/cache/visit 断言不一致；
- 任意受禁字段进入 image model 或主 readout；
- loss/gradient/parameter 出现 NaN/Inf 且一次降低 AMP 风险或学习率的诊断重跑仍未解决；
- EMA target 未更新、checkpoint 无法无损加载、validation 结果不可复现；
- M2 target/mask 错位，或无 radiomics 患者被静默丢弃；
- 输出目录已有不同 config/hash 的结果，存在覆盖风险。

### 15.3 Early stopping 与阶段门控

共同 early stopping 为最多 12 epoch、patience 4。M0 监控 validation state loss；pilot 显示 EMA latent 尺度随训练变化后 raw delta/state loss 跨 epoch 不可直接比较，因此正式 M1/M2 监控只由影像得到、基于误差和而非逐小分母平均的 raw aggregate transition gain，并同步报告 native LayerNorm aggregate gain。随机投影 SIGReg 和 M2 radiomics 项不进入 best-epoch 排序，但跨患者 feature-wise std 必须通过 eligibility 门槛。这样 M2 不会由 paired subset 决定全部患者的训练长度，也不会使用 pCR/test。最优 epoch、最后 epoch 和选择原因均保存。

若 M1 数值不稳定或 validation median transition gain 不优于 M0，不进入 M3，也不堆叠更复杂 loss；先检查 EMA/online 坐标、delta scale、LayerNorm、`alpha_delta`、target delta norm 与 copy 难度。M2 仍可在稳定的 M1 上按预定第一轮运行，以检验辅助监督，但若 `L_rad` 不优于 mean predictor、head 预测近常数、时间点错位或 image validation loss 明显恶化，则停止 lambda 扩展并进行原因分析。若 M2 image-only AUROC 未改善，必须报告负结果并检查 target 噪声、complete-case bias、loss 权重和 FTV/LD 与 ROI geometry 的重复性，不得转而用 C1 fusion 支撑主张。

M3、7 通道敏感性和数据效率实验都是可停止项；计算预算紧张时按此顺序删除，不削减 M0/M1/M2 五折、OOF prediction、主要 shortcut 或 C0/C1/C2。

## 16. 最终结果表格模板

以下主表已由正式 test/OOF 结果替换；完整 95% CI、grounding、shortcut 和 controls 见 `reports/final_report.md` 与 `metrics/final/final_analysis_v2/`。

### 16.1 Image-only pCR 主结果（完整 808 cohort）

| 模型 | 输入声明 | decision point | fold AUROC mean±std | pooled OOF AUROC | fold AUPRC mean±std | pooled OOF AUPRC | Accuracy | Sensitivity | Specificity | N |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | ROI 辅助 MRI-only | T0 | 0.5385±0.0426 | 0.5492 | 0.3723±0.0387 | 0.3702 | 0.5012 | 0.5782 | 0.4615 | 808 |
| M0 | ROI 辅助 MRI-only | T0–T1 | 0.5109±0.0283 | 0.5041 | 0.3527±0.0256 | 0.3434 | 0.4715 | 0.6364 | 0.3865 | 808 |
| M0 | ROI 辅助 MRI-only | T0–T2 | 0.5613±0.0257 | 0.5486 | 0.4313±0.0503 | 0.4050 | 0.5074 | 0.6582 | 0.4296 | 808 |
| M1 | ROI 辅助 MRI-only | T0 | 0.5243±0.0269 | 0.5164 | 0.3761±0.0219 | 0.3561 | 0.4839 | 0.5855 | 0.4315 | 808 |
| M1 | ROI 辅助 MRI-only | T0–T1 | 0.5369±0.0354 | 0.5386 | 0.3766±0.0244 | 0.3722 | 0.5396 | 0.4582 | 0.5816 | 808 |
| M1 | ROI 辅助 MRI-only | T0–T2 | 0.5372±0.0281 | 0.5405 | 0.3972±0.0300 | 0.3795 | 0.4790 | 0.6691 | 0.3809 | 808 |
| M2/C2 | 训练期 radiomics、推理期 ROI辅助 MRI-only | T0 | 0.5214±0.0273 | 0.5083 | 0.3722±0.0260 | 0.3485 | 0.4839 | 0.6000 | 0.4240 | 808 |
| M2/C2 | 训练期 radiomics、推理期 ROI辅助 MRI-only | T0–T1 | 0.5350±0.0394 | 0.5292 | 0.3750±0.0259 | 0.3627 | 0.5322 | 0.4909 | 0.5535 | 808 |
| M2/C2 | 训练期 radiomics、推理期 ROI辅助 MRI-only | T0–T2 | 0.5344±0.0282 | 0.5406 | 0.3883±0.0225 | 0.3741 | 0.4691 | 0.7345 | 0.3321 | 808 |

### 16.2 Transition 与 copy-current

| 模型 | transition | N | model normalized error | copy error | gain median [IQR] | gain > 0 比例 | predicted delta norm | target delta norm | delta cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | T0→T1 | — | — | — | — | — | — | — | — |
| M0 | T1→T2 | — | — | — | — | — | — | — | — |
| M0 | T2→T3 | — | — | — | — | — | — | — | — |
| M1 | T0→T1 | — | — | — | — | — | — | — | — |
| M1 | T1→T2 | — | — | — | — | — | — | — | — |
| M1 | T2→T3 | — | — | — | — | — | — | — | — |
| M2 | T0→T1 | — | — | — | — | — | — | — | — |
| M2 | T1→T2 | — | — | — | — | — | — | — | — |
| M2 | T2→T3 | — | — | — | — | — | — | — | — |

### 16.3 Radiomics grounding（paired OOF）

| 模型/头 | feature | transition | N | MAE | RMSE | Spearman | Pearson | R² | 方向准确率 | 预测/目标方差比 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 原生 head | FTV | T0→T1 | — | — | — | — | — | — | — | — |
| M2 原生 head | sphericity | T0→T1 | — | — | — | — | — | — | — | — |
| M2 原生 head | LD | T0→T1 | — | — | — | — | — | — | — | — |
| M2 原生 head | BPE | T0→T1 | — | — | — | — | — | — | — | — |

其余 transition 与 M0/M1/M2 统一 probe 按同一 schema 完整输出到 CSV；Markdown 可汇总主要结果，不能省略底层 prediction。

### 16.4 Shortcut 诊断

| 模型 | decision point | 条件 | OOF AUROC | 相对 native ΔAUROC | OOF AUPRC | 概率绝对变化 | transition gain | N |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M0 | T0–T1 | native | — | 0 | — | 0 | — | — |
| M0 | T0–T1 | repeated-T0 | — | — | — | — | — | — |
| M1 | T0–T2 | temporal shuffle | — | — | — | — | — | — |
| M2 | T0–T2 | temporal shuffle | — | — | — | — | — | — |

实际表必须覆盖 M0/M1/M2 的所有适用 perturbation 与 decision point。

### 16.5 控制组（同一 radiomics subset）

| 控制组 | 推理输入 | decision point | pooled OOF AUROC | pooled OOF AUPRC | N | 可支持的结论 |
|---|---|---|---:|---:|---:|---|
| C0 | 可观察 radiomics | T0/T0–T1/T0–T2 | — | — | — | radiomics 自身预测力 |
| C1 | image + 可观察 radiomics | T0/T0–T1/T0–T2 | — | — | — | 依赖表格的 fusion 上界 |
| C2 | M2 image-only | T0/T0–T1/T0–T2 | — | — | — | 主要方法，允许支持主张 |

### 16.6 阶段结论登记

| 问题 | 预注册判据 | 结果 | 结论/限制 |
|---|---|---|---|
| Next-Change 是否优于 Next-State？ | M1 vs M0 的 OOF readout、state error、gain 联合判断 | normalized gain -62.21%→+6.24%；pCR不一致 | transition objective改善，临床readout未一致改善 |
| 是否明显优于 copy-current？ | gain 分布、正值比例与 paired bootstrap | normalized是；raw gain仍-2.84% | 仅在LayerNorm空间成立 |
| M2 image-only 是否改善？ | C2 vs M1/M0，完整 cohort 为主 | T0/T0–T1/T0–T2 AUROC=0.508/0.529/0.541 | 相对M1无改善 |
| 改善是否不依赖推理期 radiomics？ | C2 readout schema/provenance 审计通过 | 推理边界通过，但没有观察到改善 | 无可确认的无表格增益 |
| 是否更依赖真实 follow-up？ | repeated-T0/shuffle 与 native 联合分析 | M2 T0–T2 repeated Δ=-0.058；shuffle Δ=-0.018 | 内容依赖部分成立，顺序依赖不充分 |
| 哪些 feature 可预测？ | OOF grounding 优于 mean predictor且非恒定 | 11/12近常数；Spearman -0.091～0.053 | 没有可靠feature |
| 是否存在 complete-case bias？ | pCR/subtype/treatment/age/fold/visit 描述差异 | 已观察到 pCR 构成差异 | 需结合完整审计解释 |
| 是否运行 M3？ | 第 6.5 节门控全部满足 | validation/OOF grounding门控失败 | 按门控未运行 |

最终报告必须基于完整的 prediction-level 文件和上述联合证据回答研究主张；“训练成功”或单个 AUROC 上升不足以证明模型学习了治疗响应变化。
