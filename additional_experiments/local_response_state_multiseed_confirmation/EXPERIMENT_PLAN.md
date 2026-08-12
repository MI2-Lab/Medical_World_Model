# LOCAL Response State Multi-seed Confirmation

## 1. Git 与实验身份

- Parent branch：`feature/local-global-response-state-pilot`
- Parent commit SHA：`78ba693ad34dbb2b5a28f0476185966714bb63c5`
- Experiment branch：`feature/local-response-state-multiseed-confirmation`
- Experiment start timestamp：`2026-08-11T04:30:44Z`
- 实验目录：`additional_experiments/local_response_state_multiseed_confirmation/`

由于共享 checkout 同时被其他任务切换 branch，本实验在同一 Git repository 的独立 worktree 中执行；branch、parent SHA 和最终 commit 均不因此改变。

## 2. 目的与冻结范围

上一轮 scientific classification 为 `A. LOCAL STATE VALIDATED IN PILOT`。本轮仅确认该候选，不搜索 architecture。

唯一比较：

```text
GAP:   encoder.features[3] final spatial map -> exact global mean -> 128-D
LOCAL: encoder.features[3] final spatial map -> fixed central 64x64x64-mm
       fractional feature-cell overlap pooling -> 128-D
both:  same Linear(128,192) + LayerNorm(192) -> response state
```

LOCAL 必须复用 pilot 的正式实现及 hash-locked audited pooling；窗口、中心、坐标、fractional overlap、normalization 和 online/EMA 对称路径不变。禁止 LOCAL_GLOBAL/global branch、window sweep、lesion-mask recenter、visit-adaptive crop、learned attention、stronger encoder、new crop、LD、pCR optimization 与 clinical auxiliary loss。

## 3. 数据与训练 contract

原样复用 pilot：C1B-H DCE7、375-patient formal cohort、947-patient technical eligibility、1486 grounding-observable visits、seed-2026 patient folds、139 fold-external train-only patients、encoder、JEPA transition、EMA、SIGReg、AdamW、LR、physical batch 4、gradient accumulation 8、logical B32、gradient clipping、12 epochs、patience 4、checkpoint selection、outer-train FTV transform 和 loss-side grounding observable mask。

Grounding target 仅为 FTV，`lambda_FTV=0.25`，不得重新调参。grounded checkpoint 必须满足 selected validation state loss `<= 1.05 x` 同 seed/fold、同 architecture no-grounding baseline；否则按冻结 fallback 标记失败。test FTV、Delta-FTV 与 pCR 不参与 checkpoint selection。

## 4. Formal matrix 与配对

Arms：`GAP0`、`GAP3`、`LOCAL0`、`LOCAL3`。`0` 为 JEPA/base only；`3` 为 JEPA/base + Direct FTV。

Seeds：`2026, 3026, 4026, 5026, 6026`；folds：`0..4`。总计 `5 seeds x 5 folds x 4 arms = 100` 个全新 formal cells。上一轮 pilot 的 2026/3026 artifacts 不进入本轮 aggregate。

同 effective seed/fold 的四个 arms 共享 encoder、projector、transition、target copies、baseline response projection 初始化和 patient order；`GAP0 -> GAP3`、`LOCAL0 -> LOCAL3` 的执行顺序保证 grounded selection 使用本轮 matching baseline。GAP 与 LOCAL 的 checkpoint initialization 和随机流按 seed/fold 严格配对。

## 5. Evaluation 与统计

每个 selected frozen online pre-projector `r_t [N,4,192]` 运行 outer-fold-isolated Ridge。Static endpoints 为 T0/T1/T2/T3/macro，报告 Spearman、Pearson、natural R2、RMSE、MAE、prediction/target variance ratio 与 descriptive calibration slope。Observed dynamics endpoints 为 T0->T1、T1->T2、T2->T3、macro，报告 Spearman、Pearson、natural R2、RMSE 与 variance ratio。

Independent unit 是 training seed。每个 seed 先 pool 五个 outer-test folds 得到一个 OOF result；五个 seed 再报告 mean、median、sample SD、min/max、direction count，以及按 seed 重采样的 10,000 次 percentile bootstrap 95% CI（固定 RNG seed `20260811`）。fold-level paired effects 仅作 sensitivity，不视为独立重复。

## 6. 结果前冻结的 confirmation gate

只有以下七项全部满足，才输出 `LOCAL_MULTISEED_CONFIRMED`；否则输出 `LOCAL_MULTISEED_NOT_CONFIRMED`：

1. `LOCAL0-GAP0` static macro Spearman 在至少 4/5 seeds **严格大于** `+0.10`；
2. 上述五个 seed effects 的 mean **严格大于** `+0.10`；
3. `LOCAL0-GAP0` observed Delta-FTV macro Spearman 在至少 4/5 seeds 严格为正；
4. `LOCAL0-GAP0` static natural R2 不得系统性恶化；本轮在结果前 operationalize 为“至少 4/5 seed effects 严格为负”即系统性恶化；
5. `LOCAL3-LOCAL0` static macro Spearman 在至少 4/5 seeds 严格为正；
6. `LOCAL3-LOCAL0` observed Delta-FTV macro Spearman 在至少 4/5 seeds 严格为正；
7. `LOCAL3 vs LOCAL0` optimization safety 至少 90%，即至少 23/25 paired folds PASS；每 fold 阈值为 state-loss degradation `<=5%`。

`GAP3 vs GAP0` safety 仅作 reference。阈值是预注册 decision rule，不是 statistical significance。pCR AUROC 不得用于选择 architecture。

## 7. 必须输出与解释边界

输出中文 `reports/final_report.md`、seed-level summary、fold-level sensitivity、training safety table、static/dynamic figures，并逐项回答 brief 的十个问题。

即使确认成功，FTV grounding 只支持 tumor burden / response information，不能证明 MRI representation 已被充分利用。后续核心问题仍是 MRI 是否包含 Patient Profile（HR/HER2 等）以外的 complementary tumor phenotype information。因此本实验是 architecture confirmation，不是 image-learning question 的终点。

## 8. 完整性与 Git

正式结果完成后，验证 100/100 cells、五 seed pooled OOF、所有 table/figure/report、private artifact permissions 与 public privacy gate；比较 parent tree，确认旧 `additional_experiments/` 目录零修改。只 stage 本实验新增的可公开复现文件，commit message 固定为 `Add LOCAL response-state multi-seed confirmation`，再 push requested branch。最终报告末尾记录 branch、commit SHA 与真实 push status；失败时记录 `GITHUB_PUSH_FAILED` 和原始 error。
