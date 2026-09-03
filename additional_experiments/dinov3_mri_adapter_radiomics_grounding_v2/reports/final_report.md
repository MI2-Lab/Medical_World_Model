# DINOv3 MRI Adapter + Direct Radiomics Grounding V2 报告

## 当前结论

当前决策为 `GROUNDING_OPTIMIZATION_CONFLICT`。

完整训练预算后 paired checkpoint safety 仍失败；这不是 target feasibility failure。

## 阶段状态

| 阶段 | 状态 |
|---|---|
| 实现与 preflight | `PASS` |
| V2 ROI feasibility | `PASS` |
| PyRadiomics extraction | `COMPLETE` |
| 五折 target feasibility | `PASS` |
| DINOv3 frozen cache | `COMPLETE` |
| private SHA manifest | `COMPLETE` |
| paired smoke cell | `PASS` |
| 75-cell representation matrix | `NO_GO` |
| representation lock | `NOT_RUN` |
| outcome-blind mechanism gate | `NOT_RUN` |
| mechanism lock | `NOT_RUN` |
| pCR evaluation | `NOT_RUN` |

## ROI 与 erosion feasibility

| Gate | Coverage | 阈值 | 结果 |
|---|---:|---:|---|
| T0 | 100.0% | 90.0% | PASS |
| T1 | 92.8% | 90.0% | PASS |
| T2 | 87.2% | 85.0% | PASS |
| T3 | 75.2% | 70.0% | PASS |
| overall | 88.8% | 85.0% | PASS |

Erosion 只用于可提取子集的 symmetric stability audit；原始 target ROI 仍保持 64 voxels/3 slices。

## Early radiomics target

五折 target gate：`PASS`；T0–T2 用于 residualizer/PCA16/grounding，T3 radiomics mask 永久为 false。
每折保留 feature 数范围为 64–64。

## Representation checkpoint safety

失败 cell：`seed4026_fold4_D2`；停止前完成 49/75 个完整 state cells。
D2 在完整 12 epochs 内的最低 validation JEPA loss 为 0.06056，高于 paired D1 允许上限 0.05293 （超出 14.4%）。
State non-collapse 未失败；停止原因是 paired objective safety constraint。

## 下一步

pCR 保持锁定。先诊断 PC-wise/visit-wise transfer 与 objective 冲突，不根据未读取的 pCR 解冻 DINO backbone。

## 解释边界

- V2 不修改 DINOv3 backbone、C1B-H geometry、cohort 或 folds。
- Representation 与 mechanism 阶段不读取 pCR 或 clinical outcome。
- 只有 representation lock 与 mechanism lock 同时有效，pCR evaluator 才能打开 outcome manifest。
- 即使最终通过，也只是内部 OOF 证据，仍需独立 cohort 验证。
