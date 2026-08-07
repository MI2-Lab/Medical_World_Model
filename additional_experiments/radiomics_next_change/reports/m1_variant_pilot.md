# M1 变体选择：仅验证集 pilot

## 目的与数据边界

M1 有两个预先定义的候选：

- `m1_delta_only`：仅优化 latent delta 回归（另含共同的 SIGReg 正则）。
- `m1`：同时优化 next-state 与 latent delta（另含共同的 SIGReg 正则）。

二者均使用 fold 0 的相同训练/验证患者、相同 seed、相同 12 个 epoch、相同优化器和网络规模；本次选择没有读取 fold 0 test，也没有训练 pCR readout。正式五折只保留选中的变体。

## 验证集结果

下表为第 12 个 epoch。`raw gain` 与 `normalized gain` 都以 copy-current 为参照，正值表示优于复制当前状态。

| 候选 | raw next MSE | raw copy MSE | raw gain | normalized next MSE | normalized copy MSE | normalized gain | state loss | delta cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| delta-only | 0.129861 | 0.145529 | +10.766% | 0.584880 | 0.576846 | -1.393% | 0.614999 | 0.255713 |
| delta + state | 0.142183 | 0.136579 | -4.103% | 0.555406 | 0.576224 | +3.613% | 0.585824 | 0.076361 |

## 锁定决定

锁定 `m1`（delta + state）作为正式 M1。理由是研究的主对象是 next-state/next-change 表征，而它是两个候选中唯一在独立 feature-wise LayerNorm 的 next-state 误差上超过 copy-current 的方案，同时 state loss 更低。delta-only 在 raw 幅度和方向余弦上更好，这一反向证据不会隐藏；它说明“真实幅度拟合”和“归一化状态演化”之间存在明显张力。

该选择是模型族选择，不是对最终 test 结果的追认。正式 M1 的 checkpoint 继续只按各 fold 验证集选择，test 仅在模型锁定后一次性评估。
