# DINOv3 MRI Adapter + Direct Radiomics Grounding 报告

## 结论

当前协议决策为 `NO_GO`。

该结果来自 radiomics Stage-A feasibility gate；representation matrix 和 pCR evaluation 均未启动，因此不能解释为 DINOv3 或 grounding 无效。

## 阶段状态

| 阶段 | 状态 |
|---|---|
| 实现与 preflight | `PASS` |
| DINOv3 frozen cache | `NOT_RUN` |
| radiomics Stage-A coverage | `NO_GO` |
| 75-cell representation matrix | `NOT_RUN` |
| evaluation lock | `NOT_RUN` |
| pCR evaluation | `NOT_RUN_STAGE_A_STOP` |

## Radiomics Stage-A 实测

| Gate | 实测 coverage | 阈值 | 结果 |
|---|---:|---:|---|
| T0 | 100.0% | 90.0% | PASS |
| T1 | 92.8% | 90.0% | PASS |
| T2 | 87.2% | 90.0% | FAIL |
| T3 | 75.2% | 70.0% | PASS |
| overall | 88.8% | 85.0% | PASS |

T2 为唯一失败 gate（87.2% < 90.0%，还差 11 个有效 visits）。按预注册规则必须停止训练；不得查看 pCR 后再放宽 ROI。

## Outcome-blind threshold sensitivity

把 axial slice 下限从 3 改成 2，T2 coverage 仅从 87.2% 变为 87.5%，不能解决 gate。
保持 3 slices 时，voxel 下限必须降至 ≤35 才能达到 90%；32 voxels 时 T2 coverage 为 90.9%。这可能降低 texture radiomics 的可靠性，只能作为新协议候选，不能补救当前结果。

## 已实现的关键约束

- DINOv3 checkpoint revision 和三个 artifact SHA-256 均固定；每个 channel/slice 保存 CLS、patch mean、patch population SD，并排除 4 个 register tokens。
- 模型 forward 仅接受 `[B,4,7,32,2304]` frozen summaries；clinical、pCR、FTV、radiomics、ROI mask 和 geometry 均不在推理接口中。
- PyRadiomics target 使用 Original-only、force2D axial、binWidth 0.25，以及 first-order/GLCM/GLRLM/GLSZM/GLDM/NGTDM；shape、wavelet 和 LoG 不启用。
- radiomics feature filtering、稳定性审计、FTV/局部 mask volume/visit residualization 和 PCA16 均按 outer-train 单独拟合。
- D1–D3 结构相同并共享初始化；D2/D3 checkpoint 选择必须满足相对 paired arm 的 JEPA/FTV 5% 安全约束。
- pCR evaluator 在 `EVALUATION_LOCK.json` 生成前 fail closed；最终 fusion 使用 inner-OOF clinical+FTV logits 和 image logits，只拟合 offset alpha/beta。

## 真实数据 smoke

已验证 DINO cache `[4, 7, 32, 2304]`/`float16`，以及 radiomics `[4, 3, 651]`（651 features）。

## 下一执行顺序

1. 不启动 75-cell matrix，也不读取 pCR。先由 PI 决定是否发起一个新的、独立预注册的 ROI feasibility revision。
2. 当前 outcome-blind sensitivity 已表明 slice gate 不是瓶颈。PI 可在新协议中选择：保留 64-voxel 可靠性并把 T2 coverage gate 预注册为 85%，或采用 ≥32 voxels 后重新做完整 morphology stability audit；前者较保守，后者风险是 texture 不稳定。
3. 只有新协议在看 pCR 前锁定且重新通过 coverage/stability gates，才可复用已经实现的 DINO cache、D1–D3 training、evaluation lock 和 fusion 代码。
