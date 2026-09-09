# 双侧 MRI Context pCR Ceiling v1

## 结论

本实验使用一次性锁定的原始 bilateral DCE-MRI 输入，测试 DINOv3 全局/空间表征是否能在既有 clinical+FTV 之上提供 pCR 信息。当前决策为 **`DINO_SPATIAL_NO_CONDITIONAL_PCR_SIGNAL`**。

V6 的 exploratory state rescue 只使用 seed-2026 的冻结 states，属于 hypothesis-development OOF，不是独立 confirmatory evidence。三个 primary timings 的 AUROC 如下：

| timing | C0_192 | S0_256 | S1_256 |
|---|---:|---:|---:|
| T0 | 0.6977 | 0.6961 | 0.6946 |
| T0_T1 | 0.7402 | 0.7355 | 0.7354 |
| T0_T2 | 0.7653 | 0.7667 | 0.7638 |

## 输入与安全

- Input lock：`LOCKED`；cache：`COMPLETE`，覆盖 `375` 个 primary patients。
- 原始 bilateral DCE 经 canonical RAS 处理；72 个已知 singular sform 仅用 valid qform 内存修复，未改写原始文件。
- DINO backbone 冻结，representation extractor 不读取 pCR、clinical、FTV、ROI 或 mask。
- formal B0/B1 training 在 exploratory gate 未通过时保持锁定；没有依据结果追加 crop、morphology 或 BPE audit。

## 解释

如果 exploratory gate 未通过，结论是现有 I-SPY2 bilateral DINO input 没有足够的 conditional pCR 方向性信号，按预注册规则停止本条 DINO/world-model 主线；下一步应获取独立 cohort 或 authoritative bilateral imaging processing，而不是继续微调 adapter。若通过，才进入预注册的 fresh-seed formal matrix。
