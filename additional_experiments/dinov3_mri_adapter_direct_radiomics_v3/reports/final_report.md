# DINOv3 MRI Adapter + Direct Radiomics Grounding V3 报告

## 当前结论

当前预注册决策为 `DIRECT_RAD_WEIGHT_SCREEN_NO_GO`。

三个 direct-radiomics 权重都未产生满足 paired JEPA safety 的可冻结 checkpoint，因此 downstream candidate mechanism 指标不可评估。
fresh-seed 50-cell 矩阵未启动，pCR outcome 未读取；该结论是当前 fine-tuning objective 的 optimization/safety failure，不是 radiomics target feasibility 或 pCR 模型负结果。

## 阶段状态

| 阶段 | 状态 |
|---|---|
| V2 inheritance | `PASS` |
| preflight | `PASS` |
| one-batch gradient smoke | `PASS` |
| pilot orchestration (6 trained; 9 fail-fast skips) | `COMPLETE` |
| pilot mechanism gate | `NO_GO` |
| pilot weight lock | `NOT_RUN` |
| 50-cell fresh-seed matrix | `NOT_RUN` |
| evaluation lock | `NOT_RUN` |
| formal mechanism gate | `NOT_RUN` |
| mechanism lock | `NOT_RUN` |
| pCR evaluation / acceptance | `FAIL` |
| private SHA summary | `COMPLETE` |

## Pilot outcome-blind mechanism gate

| Arm | λ | Direct head ρ | Matched probe ρ | 相对 C0 gain | Static FTV Δρ | ΔFTV Δρ | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| R025 | 0.25 | — | — | — | — | — | FAIL |
| R050 | 0.50 | — | — | — | — | — | FAIL |
| R100 | 1.00 | — | — | — | — | — | FAIL |

固定 V2 C0 matched-probe radiomics macro Spearman 为 0.2861。

Gate 解释：
- R025: checkpoint safety FAIL；direct head、matched probe 与 FTV retention 均未评估。
- R050: checkpoint safety FAIL；direct head、matched probe 与 FTV retention 均未评估。
- R100: checkpoint safety FAIL；direct head、matched probe 与 FTV retention 均未评估。

完整运行 cells 的 paired JEPA safety：

| Arm | Fold | C0 JEPA | 允许上限 | 最低 observed JEPA | 超上限 |
|---|---:|---:|---:|---:|---:|
| R025 | 0 | 0.09033 | 0.09485 | 0.09913 | 4.5% |
| R025 | 1 | 0.08038 | 0.08440 | 0.08667 | 2.7% |
| R050 | 0 | 0.09033 | 0.09485 | 0.09831 | 3.6% |
| R050 | 1 | 0.08038 | 0.08440 | 0.08593 | 1.8% |
| R100 | 0 | 0.09033 | 0.09485 | 0.09700 | 2.3% |
| R100 | 1 | 0.08038 | 0.08440 | 0.08471 | 0.4% |

## 研究解释与下一步

- 不扩展到 50-cell confirmatory matrix，也不读取 pCR；这正是快速 Pilot gate 的资源保护作用。
- C0 matched linear probe 已达到 0.2861，说明 residual-PC 信息在未 grounding 的 state 中已经可解码；先做 fold/PC/visit-wise audit，确认该结果不是 pooled-fold 口径造成的假象。
- 在独立协议中先训练 frozen-C0 的 head-only calibrator，区分 head/target learnability 与 representation update；它只能作机制校准，不能宣称产生了新 representation。
- 若 head-only 可行，再预注册 JEPA-preserving 更新（head warm-up、adapter 极低 LR、C0 state anchoring 或 gradient surgery），仍以 matched-probe gain 和 JEPA safety 选型，不得用未读取的 pCR 调参。

## 边界

- DINOv3 backbone 永久冻结；模型 forward 仅接收冻结 DINO summaries。
- FTV objective 权重为零，FTV 仅用于 retention diagnostic 与最终 clinical+FTV baseline。
- V2 DINO cache、五折 PCA16 targets、cohort 和 folds 均 hash-bound 复用，V2 结果不被修改。
- 公开报告不包含患者标识、预测、private path 或 radiomics targets。
