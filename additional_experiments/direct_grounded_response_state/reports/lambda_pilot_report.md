# Fold-0 λ pilot 报告

## 1. 选择结论

仅使用 fold 0 的 train/validation 数据，在预注册候选 `{0.02, 0.05, 0.10, 0.25}` 中选择 `lambda_FTV = 0.25`。这是 G3 与 G4 同时通过联合有效性门槛的最小候选，因此选择模式为 `smallest_effective_lambda`，没有启用 fallback，也没有读取 test feature、test FTV、pCR label 或 test AUROC。

## 2. Validation 证据

| λ | G3−G1 macro FTV Spearman | G4−G2 macro FTV Spearman | G3 base degradation | G4 base degradation | 联合有效 |
|---:|---:|---:|---:|---:|:---:|
| 0.02 | -0.0154 | -0.0051 | -0.38% | -0.08% | 否 |
| 0.05 | -0.0223 | +0.0053 | -0.79% | -0.20% | 否 |
| 0.10 | +0.0026 | +0.0141 | -0.88% | -0.39% | 否 |
| 0.25 | +0.0327 | +0.0693 | -1.77% | -1.60% | 是 |

在 `lambda_FTV = 0.25` 时，G3 validation macro FTV Spearman 为 0.5310，对应 G1 为 0.4983；G4 为 0.5737，对应 G2 为 0.5043。两组 representation std 分别为 0.645 与 0.805，均高于 0.05 的防 collapse 门槛；base loss 相对配对 baseline 均改善而非恶化，满足不超过 5% degradation 的硬约束。

## 3. 数据与决策边界

每个候选及两组配对 baseline 都严格包含 525 名 fold-0 train 患者和 121 名 validation 患者。Ridge 的 feature scaler、模型和 target transform 只在 train 拟合；validation 每时点预测一次。选择结果、候选逐项指标、输入资产 SHA-256 与 split closure 证据保存在 `metrics/lambda_selection/`。后续 G3/G4 五折正式训练固定使用同一个 `lambda_FTV = 0.25`，不因 test 结果重新调参。
