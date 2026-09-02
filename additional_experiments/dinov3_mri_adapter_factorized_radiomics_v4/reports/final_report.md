# V4 DINOv3 MRI-domain adapter + factorized radiomics grounding：Pilot 报告

## 结论

Pilot 完成了 5 folds × 4 arms = 20 个 outcome-blind cells，但没有任何 radiomics 权重通过预注册 mechanism gate。结论为 `FACTORIZED_RADIOMICS_PILOT_NO_GO`；pCR evaluator 保持锁定。这个结果说明当前 factorized 结构在本 pilot 参数下没有把 radiomics 信号迁移到 phenotype state，不等于 DINOv3 或 MRI image 本身没有可用信息。

## 设计与安全性

- 复用 V2 hash-bound DINO summaries、375 人 fold targets 和固定 outer folds；没有重新提取 cache。
- forward 只接收冻结 DINO summary；FTV loss 权重严格为 0。state 分为 128-D JEPA branch 与 64-D phenotype branch。
- 所有 20 个 state archive 均为 `[808, 4, 192]`，I-SPY2 每 fold 恰好一次；I-SPY1 只作为 train-only。
- checkpoint、state hash、paired initialization、T3 mask 和 runtime outcome/clinical sentinel 均通过审计。

## Outcome-blind mechanism 结果

| arm | direct-head macro Spearman | matched-state probe | probe gain vs F0 | FTV static drop | FTV delta drop |
|---|---:|---:|---:|---:|---:|
| F0 | -0.0097 | 0.2369 | 0.0000 | 0.0000 | 0.0000 |
| F005 | 0.0025 | 0.2382 | 0.0013 | 0.0026 | -0.0010 |
| F010 | 0.0042 | 0.2381 | 0.0011 | 0.0023 | -0.0005 |
| F025 | 0.0098 | 0.2376 | 0.0007 | 0.0028 | 0.0018 |

F0 的 matched probe 已有约 0.237 的 radiomics macro Spearman；candidate 只增加约 0.001 左右，远低于预设的 +0.05。direct head 的绝对相关也只有约 0.003–0.010，未达到 0.10。候选训练后续 epoch 的 validation radiomics loss 确实继续下降，但同时 JEPA loss 超过 paired F0 的 105% safety ceiling，因此正式 checkpoint 只能选择早期安全 epoch；这直接提示当前共享 adapter/transition 优化仍存在冲突。FTV static/Δ diagnostics 没有明显下降，因此本轮主要失败点是 grounding transfer 不足，而不是 JEPA/FTV retention 损坏。

## 下一步

按照预注册停止规则，不运行正式 50-cell matrix，也不打开 pCR。下一轮若继续，应先做低成本诊断：检查 phenotype branch 的 target scale、direct-head optimization/selection 和 probe ceiling；然后以 detached auxiliary representation loss 或更强的 phenotype-only pretraining 做小规模 ablation，并预先固定新的 gate。只有出现稳定的 held-out phenotype transfer，才值得重新进入 pCR 评估。

本报告仅为内部 representation study，不能作临床或独立泛化声明。
