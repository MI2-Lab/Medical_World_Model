# M2 fold 0 Lambda 选择报告

生成时间（UTC）：`2026-08-06T17:38:40.989087+00:00`

## 结论

按预注册规则锁定 **`lambda_rad=0.05`**（run `m2_lambda005`）。

该结论只使用 fold 0 validation。脚本没有提取 test split、没有加载 test DCE array，
也没有把 pCR 标签用于候选排序。`load_evaluation` 会核验 checkpoint 中锁定的
test patient ID/hash 元数据，但这不读取 test 图像内容或计算任何 test 指标。

## 预注册选择规则

1. 候选固定为 `lambda_rad={0.05, 0.1, 0.25, 0.5}`，且必须与同一 fold 0 M1
   使用相同 train/validation IDs、transform、模型规格、训练配置与实现哈希。
2. 任一候选 best epoch 的 `val_normalized_next_mse` 或 `val_state_loss` 相对 M1
   恶化超过 5%，即排除。raw/normalized aggregate gain 均记录，但不反向改变规则。
3. history/prediction 非有限、跨患者 latent feature std 低于训练 eligibility 门槛、
   predicted-delta norm/方差在数值精度下为零，或任一 radiomics feature 的
   validation 预测方差小于等于 `1e-12`，均视为数值/坍塌失败。
4. 在剩余候选中选择 paired validation standardized radiomics MAE 最低者；若 MAE
   位于最小值的 1% 以内，取更小的 lambda。Spearman 和 prediction variance 只作
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

## 各 Feature Validation Grounding

所有指标均在对应 fold-train-only transform 的 standardized 空间计算；每名配对患者
贡献 T0→T1、T1→T2、T2→T3 三个有效 transition。

| lambda | feature | N | MAE | RMSE | Spearman | prediction variance | target variance | 方差比 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | ftv | 177 | 0.5851 | 0.7412 | 0.0672 | 0.005107 | 0.549672 | 0.0093 |
| 0.05 | sphericity | 177 | 0.8268 | 1.1810 | -0.0235 | 0.005610 | 1.392290 | 0.0040 |
| 0.05 | ld | 177 | 1.2685 | 2.3095 | 0.0494 | 0.004198 | 4.835562 | 0.0009 |
| 0.05 | bpe | 177 | 0.7620 | 1.0917 | -0.0177 | 0.003271 | 1.185588 | 0.0028 |
| 0.1 | ftv | 177 | 0.5841 | 0.7412 | 0.0709 | 0.006466 | 0.549672 | 0.0118 |
| 0.1 | sphericity | 177 | 0.8269 | 1.1796 | 0.0002 | 0.006038 | 1.392290 | 0.0043 |
| 0.1 | ld | 177 | 1.2669 | 2.3084 | 0.0621 | 0.004311 | 4.835562 | 0.0009 |
| 0.1 | bpe | 177 | 0.7603 | 1.0913 | -0.0027 | 0.003415 | 1.185588 | 0.0029 |
| 0.25 | ftv | 177 | 0.5841 | 0.7419 | 0.0616 | 0.010512 | 0.549672 | 0.0191 |
| 0.25 | sphericity | 177 | 0.8299 | 1.1767 | 0.0368 | 0.008239 | 1.392290 | 0.0059 |
| 0.25 | ld | 177 | 1.2655 | 2.3018 | 0.0858 | 0.005493 | 4.835562 | 0.0011 |
| 0.25 | bpe | 177 | 0.7573 | 1.0915 | 0.0249 | 0.004462 | 1.185588 | 0.0038 |
| 0.5 | ftv | 177 | 0.5868 | 0.7449 | 0.0435 | 0.017773 | 0.549672 | 0.0323 |
| 0.5 | sphericity | 177 | 0.8333 | 1.1751 | 0.0798 | 0.012605 | 1.392290 | 0.0091 |
| 0.5 | ld | 177 | 1.2656 | 2.2926 | 0.1029 | 0.008674 | 4.835562 | 0.0018 |
| 0.5 | bpe | 177 | 0.7564 | 1.0916 | 0.0515 | 0.006324 | 1.185588 | 0.0053 |

## Test-Blind 审计与可复现性

- `extract_native_split` 调用记录：`[{'run_name': 'm2_lambda005', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda01', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda025', 'split': 'val', 'patient_count': 121}, {'run_name': 'm2_lambda05', 'split': 'val', 'patient_count': 121}]`；
  每次 split 均严格为 `val`。
- test DCE arrays accessed：`false`；test metrics computed：`false`；
  pCR used for selection：`false`。
- API 返回对象包含未被本脚本读取或用于计算的 validation pCR label array；候选排序
  仅依赖 image/radiomics validation 指标。
- 选择脚本 SHA-256：`da128e5de1a1015d43ddf692fc783915bad4f4cb0f6e8522a7dac8dd7708fe1f`。checkpoint、config 和 history
  的绝对路径与 SHA-256 逐候选保存在 JSON/CSV 中。
- 三个正式输出均采用 create-new 写入；目标已存在时脚本在评估前拒绝覆盖。
