# M2 fold 0 Lambda 选择报告

生成时间（UTC）：`2026-08-06T17:46:28.670683+00:00`

## 结论

按预注册规则锁定 **`lambda_rad=0.05`**（run `m2_lambda005`）。

该结论只使用 fold 0 validation。脚本没有提取 test split、没有加载 test DCE array，
也没有把 pCR 标签用于候选排序。必须准确说明：`load_evaluation` 为执行锁定契约会
物理加载 cohort metadata、全 cohort pCR labels 与 radiomics raw metadata，并核验
train/val/test patient ID/hash；这些 metadata/labels 绝不进入候选排序。只有 val DCE
array 经 DataLoader 进入模型，也没有计算任何 test 指标。

## 预注册选择规则

1. 候选固定为 `lambda_rad={0.05, 0.1, 0.25, 0.5}`，且必须与同一 fold 0 M1
   使用相同 train/validation IDs、transform、模型规格、训练配置与实现哈希。
2. 任一候选 best epoch 的 `val_normalized_next_mse` 或 `val_state_loss` 相对 M1
   恶化超过 5%，即排除。raw/normalized aggregate gain 均记录，但不反向改变规则。
3. history/prediction 非有限、跨患者 latent feature std 低于训练 eligibility 门槛、
   predicted-delta norm/方差在数值精度下为零，或任一 radiomics feature 的
   任一 transition×feature validation 预测方差小于等于 `1e-12`，或 image/radiomics
   shared/head 首批 gradient 非有限或为零，均视为数值/坍塌失败；不设置无依据上限。
4. 在剩余候选中选择 paired validation standardized radiomics MAE 最低者；若 MAE
   与最小值的相对差严格小于 1%，取更小的 lambda。Spearman 和 prediction variance 只作
   grounding/坍塌诊断，不用 pCR 或 test 打破平局。

## M1 参照

- run：`m1_final`；best epoch：12；checkpoint SHA-256：
  `c9b4f3a854c33bd7283e931912696e179fc6e466b2a4c91c905009825170d20d`
- raw aggregate gain：-0.0410；
  native normalized aggregate gain：0.0361
- normalized next MSE：0.5554；
  state loss：0.5858；
  delta loss：0.0721

## 候选汇总

| lambda | best epoch | 通过筛选 | standardized MAE | RMSE | normalized MSE 相对 M1 | state 相对 M1 | raw gain | normalized gain | weighted grad ratio | 原因/结论 |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 12 | 是 | 0.8606 | 1.4552 | -0.0021 | -0.0021 | -0.0415 | 0.0361 | 0.1134 | 选中 |
| 0.1 | 12 | 是 | 0.8595 | 1.4543 | -0.0039 | -0.0040 | -0.0418 | 0.0354 | 0.2396 | 通过，但未按 MAE/1% 小 lambda 规则选中 |
| 0.25 | 12 | 是 | 0.8592 | 1.4513 | -0.0061 | -0.0066 | -0.0431 | 0.0326 | 0.6810 | 通过，但未按 MAE/1% 小 lambda 规则选中 |
| 0.5 | 12 | 是 | 0.8605 | 1.4477 | -0.0090 | -0.0097 | -0.0477 | 0.0276 | 1.5737 | 通过，但未按 MAE/1% 小 lambda 规则选中 |

表中两个“相对 M1”字段是 loss 的相对变化，正值表示恶化；5% 对应 `0.05`。
weighted grad ratio 是 best epoch 首个 training batch 的加权 radiomics shared-gradient
norm 与 image-task gradient norm 之比，不是 validation/test 调参信号。

## 各 Transition × Feature Validation Grounding

所有指标均在对应 fold-train-only transform 的 standardized 空间计算；每名配对患者
贡献 T0→T1、T1→T2、T2→T3 三个有效 transition。

| lambda | transition | feature | N | MAE | RMSE | Spearman | prediction variance | target variance | 方差比 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | T0→T1 | ftv | 59 | 0.4669 | 0.5915 | 0.1001 | 0.003009 | 0.343663 | 0.0088 |
| 0.05 | T0→T1 | sphericity | 59 | 0.6475 | 0.8239 | -0.2200 | 0.003785 | 0.612207 | 0.0062 |
| 0.05 | T0→T1 | ld | 59 | 0.4222 | 0.6113 | 0.0094 | 0.003501 | 0.370669 | 0.0094 |
| 0.05 | T0→T1 | bpe | 59 | 0.8481 | 1.2510 | -0.0587 | 0.001597 | 1.497426 | 0.0011 |
| 0.05 | T1→T2 | ftv | 59 | 0.6715 | 0.8262 | -0.0321 | 0.005103 | 0.662481 | 0.0077 |
| 0.05 | T1→T2 | sphericity | 59 | 0.8281 | 1.2303 | 0.1715 | 0.006152 | 1.537178 | 0.0040 |
| 0.05 | T1→T2 | ld | 59 | 1.7145 | 2.8550 | 0.2437 | 0.004662 | 6.581982 | 0.0007 |
| 0.05 | T1→T2 | bpe | 59 | 0.6327 | 0.9096 | -0.0023 | 0.003618 | 0.821894 | 0.0044 |
| 0.05 | T2→T3 | ftv | 59 | 0.6170 | 0.7847 | 0.0133 | 0.006016 | 0.605165 | 0.0099 |
| 0.05 | T2→T3 | sphericity | 59 | 1.0048 | 1.4115 | -0.1423 | 0.006342 | 1.914550 | 0.0033 |
| 0.05 | T2→T3 | ld | 59 | 1.6686 | 2.7345 | -0.2353 | 0.003872 | 6.551453 | 0.0006 |
| 0.05 | T2→T3 | bpe | 59 | 0.8051 | 1.0877 | -0.0421 | 0.004223 | 1.148789 | 0.0037 |
| 0.1 | T0→T1 | ftv | 59 | 0.4614 | 0.5862 | 0.1870 | 0.004066 | 0.343663 | 0.0118 |
| 0.1 | T0→T1 | sphericity | 59 | 0.6454 | 0.8234 | -0.1793 | 0.003783 | 0.612207 | 0.0062 |
| 0.1 | T0→T1 | ld | 59 | 0.4190 | 0.6099 | 0.0099 | 0.003439 | 0.370669 | 0.0093 |
| 0.1 | T0→T1 | bpe | 59 | 0.8486 | 1.2528 | -0.0881 | 0.001449 | 1.497426 | 0.0010 |
| 0.1 | T1→T2 | ftv | 59 | 0.6711 | 0.8269 | -0.0474 | 0.006321 | 0.662481 | 0.0095 |
| 0.1 | T1→T2 | sphericity | 59 | 0.8300 | 1.2310 | 0.1710 | 0.006504 | 1.537178 | 0.0042 |
| 0.1 | T1→T2 | ld | 59 | 1.7134 | 2.8534 | 0.2515 | 0.004480 | 6.581982 | 0.0007 |
| 0.1 | T1→T2 | bpe | 59 | 0.6299 | 0.9091 | 0.0057 | 0.003695 | 0.821894 | 0.0045 |
| 0.1 | T2→T3 | ftv | 59 | 0.6198 | 0.7878 | -0.0270 | 0.007227 | 0.605165 | 0.0119 |
| 0.1 | T2→T3 | sphericity | 59 | 1.0052 | 1.4075 | -0.1332 | 0.006708 | 1.914550 | 0.0035 |
| 0.1 | T2→T3 | ld | 59 | 1.6683 | 2.7334 | -0.2306 | 0.004062 | 6.551453 | 0.0006 |
| 0.1 | T2→T3 | bpe | 59 | 0.8023 | 1.0848 | -0.0245 | 0.004386 | 1.148789 | 0.0038 |
| 0.25 | T0→T1 | ftv | 59 | 0.4542 | 0.5782 | 0.1822 | 0.007285 | 0.343663 | 0.0212 |
| 0.25 | T0→T1 | sphericity | 59 | 0.6449 | 0.8254 | -0.1324 | 0.004648 | 0.612207 | 0.0076 |
| 0.25 | T0→T1 | ld | 59 | 0.4161 | 0.6099 | 0.0133 | 0.003776 | 0.370669 | 0.0102 |
| 0.25 | T0→T1 | bpe | 59 | 0.8504 | 1.2559 | -0.0705 | 0.001886 | 1.497426 | 0.0013 |
| 0.25 | T1→T2 | ftv | 59 | 0.6702 | 0.8277 | -0.0623 | 0.009562 | 0.662481 | 0.0144 |
| 0.25 | T1→T2 | sphericity | 59 | 0.8393 | 1.2333 | 0.1074 | 0.008072 | 1.537178 | 0.0053 |
| 0.25 | T1→T2 | ld | 59 | 1.7112 | 2.8440 | 0.2458 | 0.004790 | 6.581982 | 0.0007 |
| 0.25 | T1→T2 | bpe | 59 | 0.6247 | 0.9121 | 0.0158 | 0.004455 | 0.821894 | 0.0054 |
| 0.25 | T2→T3 | ftv | 59 | 0.6278 | 0.7948 | -0.0826 | 0.010422 | 0.605165 | 0.0172 |
| 0.25 | T2→T3 | sphericity | 59 | 1.0054 | 1.3971 | -0.0748 | 0.008433 | 1.914550 | 0.0044 |
| 0.25 | T2→T3 | ld | 59 | 1.6692 | 2.7266 | -0.1777 | 0.005131 | 6.551453 | 0.0008 |
| 0.25 | T2→T3 | bpe | 59 | 0.7966 | 1.0795 | 0.0246 | 0.005006 | 1.148789 | 0.0044 |
| 0.5 | T0→T1 | ftv | 59 | 0.4490 | 0.5722 | 0.1642 | 0.013247 | 0.343663 | 0.0385 |
| 0.5 | T0→T1 | sphericity | 59 | 0.6406 | 0.8248 | -0.0749 | 0.006366 | 0.612207 | 0.0104 |
| 0.5 | T0→T1 | ld | 59 | 0.4160 | 0.6126 | -0.0007 | 0.005163 | 0.370669 | 0.0139 |
| 0.5 | T0→T1 | bpe | 59 | 0.8493 | 1.2531 | 0.0674 | 0.002811 | 1.497426 | 0.0019 |
| 0.5 | T1→T2 | ftv | 59 | 0.6752 | 0.8327 | -0.0927 | 0.014928 | 0.662481 | 0.0225 |
| 0.5 | T1→T2 | sphericity | 59 | 0.8536 | 1.2374 | 0.0836 | 0.010966 | 1.537178 | 0.0071 |
| 0.5 | T1→T2 | ld | 59 | 1.7108 | 2.8315 | 0.2143 | 0.006566 | 6.581982 | 0.0010 |
| 0.5 | T1→T2 | bpe | 59 | 0.6233 | 0.9198 | -0.0029 | 0.005939 | 0.821894 | 0.0072 |
| 0.5 | T2→T3 | ftv | 59 | 0.6361 | 0.8023 | -0.1116 | 0.015837 | 0.605165 | 0.0262 |
| 0.5 | T2→T3 | sphericity | 59 | 1.0056 | 1.3895 | -0.0058 | 0.011593 | 1.914550 | 0.0061 |
| 0.5 | T2→T3 | ld | 59 | 1.6699 | 2.7158 | -0.0813 | 0.007866 | 6.551453 | 0.0012 |
| 0.5 | T2→T3 | bpe | 59 | 0.7965 | 1.0762 | 0.0320 | 0.005665 | 1.148789 | 0.0049 |

## Test-Blind 审计与可复现性

- `extract_native_split` 调用记录：`[{'run_name': 'm2_lambda005', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda01', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda025', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda05', 'split': 'val', 'patient_count': 121}]`；
  每次 split 均严格为 `val`。
- test DCE arrays accessed：`false`；test metrics computed：`false`；
  pCR used for selection：`false`。
- `load_evaluation` 为锁定契约物理读取 cohort metadata、pCR label metadata 与
  radiomics raw metadata；`extract_native_split` 返回 validation pCR label array。
  selector 不读取 label 数值且排序仅依赖 image/radiomics validation 指标。
- 每个 transform 在 selector 内用锁定 fold-train IDs 独立重拟合，并逐字段比较
  version、raw/train hash、quantile 与全部 feature 参数；validation target 仅对
  extracted val patient IDs 逐人调用 `transform_one`，从未 `transform_all(375人)`。
- 选择脚本 SHA-256：`bc7921d937ecfe08f6d7d6bc3b38323e3a19d428b4e3c750ebfdc9dac7be1b21`。checkpoint、config 和 history
  的绝对路径与 SHA-256 逐候选保存在 JSON/CSV 中。
- 三个正式输出均采用 create-new 写入；目标已存在时脚本在评估前拒绝覆盖。
