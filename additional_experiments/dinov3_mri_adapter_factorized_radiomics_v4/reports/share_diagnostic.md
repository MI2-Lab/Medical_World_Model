# V4 Shared-gradient diagnostic

## 结论

五折 outcome-blind 诊断支持 shared adapter 梯度冲突假设：shared adapter 的 median fold JEPA/radiomics gradient cosine 为 `-0.2036`，五折中 `4/5` 个 fold 的 median cosine 为负，fold-level negative fraction 平均为 `0.8000`。同时，冻结 F0 representation、只训练 radiomics head 后 validation Smooth-L1 平均从 `0.5368` 降至 `0.4217`。

这说明 radiomics target 和线性 head 本身可学习；V4 的主要问题是 radiomics 更新经过 shared adapter/response projection 时，与 JEPA 更新方向相冲突。该诊断不读取 pCR，也不改变 V4 的预注册 NO-GO 决策。

## 五折聚合结果

| module group | JEPA grad norm | radiomics grad norm | median fold cosine | negative fraction mean |
|---|---:|---:|---:|---:|
| shared_adapter | 3.9451 | 2.0496 | -0.2036 | 0.8000 |
| response_projection | 0.7841 | 0.8609 | -0.0516 | 0.6000 |
| phenotype_branch | 0.5365 | 1.0423 | -0.0214 | 0.5333 |
| radiomics_head | 0.0000 | 0.5501 | — | — |

## 下一步

下一轮应使用真正独立的 radiomics adapter：冻结或保持 C0 的 JEPA path，另建只从 DINO summary 输入的 trainable radiomics path；radiomics loss 不得回传到 JEPA adapter。先做小规模 five-fold mechanism pilot，通过后才考虑 formal matrix 或 pCR。
