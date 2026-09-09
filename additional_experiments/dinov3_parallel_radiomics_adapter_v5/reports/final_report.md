# DINOv3 Parallel Radiomics Adapter V5 Pilot 报告

## 结论

V5 成功实现了真正的 parallel radiomics adapter：radiomics branch 的训练梯度不进入 C0，C0 state 在训练前后保持一致。5-fold outcome-blind pilot 的 direct head 和 radiomics representation 均有信号，但 matched-probe gain 为 `+0.0345`，低于预设 `+0.05`，因此决策为 `PARALLEL_ADAPTER_NOT_TRANSFERRED`。formal 50-cell 和 pCR evaluation 保持锁定。

## Pilot 结果

| metric | value |
|---|---:|
| initial/head-only matched probe | 0.2369 |
| trained parallel matched probe | 0.2714 |
| matched-probe gain | 0.0345 |
| direct radiomics head | 0.2309 |
| positive gain folds | 4/5 |
| FTV static change | 0.0136 |
| FTV delta change | 0.0115 |

## 解释

- Head-only validation loss 在五个 fold 都下降，说明 target/head 可学习。
- 独立 adapter 训练后 matched probe 平均从约 0.237 提升到约 0.271，说明 parallel 结构比 V4 shared 结构确实能产生更多 radiomics representation。
- 但 gain 尚未达到 +0.05，且 fold 之间仍有异质性；不能据此解锁 formal 或 pCR。
- C0 state identity、256-D state contract、radiomics梯度隔离、finite/noncollapse 和 privacy checks 均通过。

## 下一步

保持当前 V5 pilot 结果不可变。若继续研究，应在新独立协议中预先选择一种改进：增加 radiomics branch 的容量、使用冻结 C0 state 的 residual radiomics target，或单独训练 rad temporal transition；不能在本分支事后调整 gate、学习率或 seed。任何新协议仍需先通过 outcome-blind mechanism gate。
