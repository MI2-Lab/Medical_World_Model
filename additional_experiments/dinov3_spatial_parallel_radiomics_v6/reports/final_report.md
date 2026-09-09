# DINOv3 Spatial Parallel Radiomics Adapter V6：Pilot 报告

## 结论

V6 pilot 按预注册 gate 停止，结论为 `NO_GO`，原因是 morphology-specific spatial gain 未达到预设阈值。该结果不是“spatial branch 完全无效”：S1 相对 S0 的整体 matched-probe 增益为 `+0.0468`，并且 direct radiomics head、texture、intensity 和 kinetic family 均通过对应的 pilot 条件；但是 morphology family 的增益只有 `+0.0010`，低于要求的 `+0.03`。

因此没有启动 fresh-seed formal matrix，也没有读取 pCR 或运行任何 pCR evaluator。`EVALUATION_LOCK.json`、`MECHANISM_LOCK.json` 和 `PILOT_LOCK.json` 均不应生成。

## 已完成工作

- 建立 V6 独立目录和 protocol，继承固定 V2/V4/V5 cohort、fold、DINOv3 revision 和 target contract。
- 完成 947 人 spatial DINO cache：`[4,7,32,7,7,768]`，947/947，无缺失，float16、finite、contract 和 ordered hash 均通过。
- 完成四个 family、T0–T2、C0-residual PCA4 target；target feasibility 通过。
- 完成 preflight 和 gradient-isolation smoke：C0 path 无梯度，radiomics/spatial branch 有非零梯度，T3 target loss 为零，target/mask 只影响 loss，不影响 forward state。
- 完成 pilot 的 5 folds × 2 arms = 10 cells，并导出 states。

## Pilot 结果

| 指标 | 结果 | gate | 状态 |
|---|---:|---:|---|
| S1 direct-head macro Spearman | 0.4708 | ≥0.15 | 通过 |
| S1 matched-probe macro Spearman | 0.3118 | ≥0.15 | 通过 |
| S1 − P0_INIT probe gain | +0.1835 | ≥0.05 | 通过 |
| S1 − S0_SUMMARY probe gain | +0.0468 | ≥0.03 | 通过 |
| positive folds vs S0 | 4/5 | ≥4/5 | 通过 |
| morphology absolute probe | 0.1063 | ≥0.10 | 通过 |
| morphology S1 − S0 | +0.0010 | ≥+0.03 | **失败** |
| texture absolute probe | 0.4390 | ≥0.10 | 通过 |
| texture S1 − S0 | +0.0577 | ≥+0.03 | 通过 |
| intensity S1 − S0 | +0.0687 | ≥−0.02 | 通过 |
| kinetic S1 − S0 | +0.0599 | ≥−0.02 | 通过 |
| FTV/static retention | −0.0075 vs initial | ≥−0.02 | 通过 |

Fold-level S1−S0 matched-probe gains 为：`+0.0489, +0.0311, +0.0855, −0.0490, +0.1174`。这说明 spatial branch 的总体增益并非单一 fold 偶然，但 morphology 增益不稳定：对应 fold-level morphology probe 为 `0.1945, 0.0951, 0.1106, 0.0685, 0.0628`，而 S0 为 `0.1086, 0.1351, 0.0764, 0.1493, 0.0572`。

## 解释

V6 支持两个较窄的判断：

1. 保留 7×7 patch 空间位置后，独立 radiomics branch 能把更多 non-FTV radiomics signal 写入 64-D `z_RAD`；这个增益主要来自 intensity、kinetic 和 texture family。
2. 当前 7×7 spatial adapter 尚未可靠地学习 lesion morphology 的额外信息。morphology 的绝对可解码性勉强超过 0.10，但相对于 summary-only S0 几乎没有增量，因此不能声称 spatial tokens 已解决 V5 的 morphology/shape 信息缺失。

这也解释了为什么不能继续 formal 或解锁 pCR：预注册目标要求四个 family 中 morphology 与 texture 都有空间增量；如果现在继续跑 75 cells 或打开 pCR，会把一个已知机制 gate 失败的 representation 带入 outcome 分析，无法回答“空间信息是否真正转移”的因果问题。

## 工程/安全检查

- 947 人 spatial cache 完整；source/contract hash 已记录。
- DINO backbone 和 C0 path 冻结；C0 gradient 为零。
- S1 spatial projector、Conv/fusion、adapter、branch/head 的训练梯度检查通过。
- C0/S1 initial state 的 CUDA ulp 差异最大约 `3.04e-6`；已将 S1 frozen/initial arrays 绑定到同 fold 的 paired S0 reference，并记录在 `metrics/initial_state_canonicalization.json`。
- 所有 representation artifacts 的 `outcome_fields_read` 和 `clinical_fields_read` 为空。
- pCR 状态保持锁定；未生成 formal checkpoint 或 pCR 输出。

## 下一步建议

V6 不建议原样扩展 formal。最有效的后续工作应是先做一个小规模、outcome-blind 的 morphology diagnostics，而不是继续增加 seeds：

1. 审计 morphology target 本身：比较 workbook LD/sphericity 与 LOCAL mask elongation/flatness 的重复性、缺失模式和与 C0/S0 的残差方差，确认 morphology residual 是否还有可学习 signal。
2. 对 S1 做 spatial attribution/ablation：分别去掉 7×7 token、只保留 lesion-central tokens、只保留边界环，检查 morphology residual 是否真的依赖边界信息。
3. 如果 morphology residual 可行但 S1 仍无增量，下一轮只改 morphology branch 的空间 inductive bias（例如 boundary-aware pooling 或低维 shape summary），不修改 backbone、cohort、pCR gate 或 formal seeds。
4. 只有 morphology spatial gain 在新的、预先写死的 pilot gate 中通过，才重新考虑 formal matrix；本次 V6 的失败结果不授权事后放宽 gate。

## 产物

- `target_feasibility.json`
- `spatial_cache_check.json`
- `inheritance_check.json`
- `isolation_check.json`
- `mechanism_gate.json`
- `acceptance_check.json`
- `decision.json`
- `manifests/private_sha_manifest.json`
- `metrics/pilot_gate.json`

本报告及上述 machine-readable artifacts 属于内部 hypothesis-development OOF 记录，不能作为独立 cohort 的 confirmatory 或临床泛化证据。
