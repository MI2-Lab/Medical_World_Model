# DINOv3 三家族 Radiomics → pCR Rescue v1

## 结论

Stage A exploratory 已完成，最终决策为 `DINO_SPATIAL_NO_CONDITIONAL_PCR_SIGNAL`。V6 的 spatial states 对三家族 radiomics 的 representation transfer 不能转化为足够的 conditional pCR 增量，因此没有启动 fresh-seed formal matrix，也不再进行 morphology audit。

## Exploratory 结果

- S1 相对 clinical+FTV 的 T0–T1 ΔAUROC：`0.0043`。
- S1 相对 clinical+FTV 的 T0–T2 ΔAUROC：`0.0019`。
- early macro ΔAUROC：`0.0031`，低于继续 formal 的 `0.01` 门槛。
- S1 相对 S0 的 early macro ΔAUROC：`0.0013`。
- S1 相对 clinical+FTV 的 patient bootstrap early effect：均值 `0.0032`，95% CI `[-0.0095, 0.0161]`。
- S1 相对 S0 的 patient bootstrap early effect：均值 `0.0014`，95% CI `[-0.0054, 0.0082]`。

## 解释

三家族 spatial radiomics representation 确实在 V6 mechanism probe 中可解码，但在最终 clinical+FTV offset fusion 中只产生接近零的增量。这个结果不支持继续投入 75-cell formal matrix；继续训练相同 adapter 只会增加 seeds，而不会解决已观察到的 representation-to-outcome translation bottleneck。

## 后续建议

1. 暂停 DINOv3 spatial adapter → pCR 这条路线的重复实验。
2. 若继续研究，应改变问题设定，而不是增加同构训练：优先考虑直接优化 response/pCR-relevant image objective，或使用 lesion-level supervised/contrastive representation 作为独立路线。
3. morphology 结果保留为 secondary failure analysis，不再作为下一轮实验的入口条件。

所有 pCR 读取均发生在 exploratory lock 之后；结果属于内部 hypothesis-development OOF，不是独立 cohort 的 confirmatory evidence。
