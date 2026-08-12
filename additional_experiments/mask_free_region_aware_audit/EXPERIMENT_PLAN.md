# Mask-Free Region-Aware Representation Audit

## 1. 目的与证据等级

本实验是 frozen-feature diagnostic，检验固定、物理定义、在新 readout 中不读取 lesion mask/FTV/clinical/pCR/future visit 的区域表示，能否部分复现 Goal 5 的 PERI20 pCR 定位信号。禁止根据结果重训 encoder、JEPA 或任何主模型，也禁止启动 attention、MIL、segmentation-guided、region-token 等 architecture intervention。

“mask-free”严格限定为**本实验新增的区域构建与 pooling 不读取 mask/bbox/label**。上游 C1B-H crop contract 已冻结，独立审计确认本 808 人 evaluation cohort 的 crop center 均来自 released T0 localization-support bbox center；因此本实验不是 acquisition-centered、端到端 mask-independent deployment 验证。任何正结果只能写成“在 frozen pre-centered C1B-H coordinate system 条件下的固定坐标 readout 信号”。

## 2. Git 与开始 provenance

- Parent branch：`feature/spatial-heterogeneity-phenotype-audit`
- Parent commit：`7742d737d92ed153b5c721cd323528b0a127d5ef`
- Experiment branch：`feature/mask-free-region-aware-audit`
- Start timestamp：`2026-08-12T01:40:27-04:00`（`2026-08-12T05:40:27Z`）
- 新增文件只允许位于 `additional_experiments/mask_free_region_aware_audit/`。

## 3. 冻结上游 contract

冻结 C1B-H/DCE7、808 eligible cohort、5 folds、LOCAL0/LOCAL3、seed 2026/3026、20 个 test-blind selected checkpoints、online encoder、JEPA、FTV grounding、固定 64-mm LOCAL support、response-state selection、clinical/FTV timing。只改变 frozen online encoder final pre-pooling map 的区域 readout。

输入为 `[B,4,7,112,176,160]` float32；final map 的空间 shape 必须从 runtime tensor 动态读取，并由冻结卷积 geometry 验证（正式环境预计 `[B*4,128,14,22,20]`）。不得持久化 raw spatial map。

## 4. 禁止信息与因果 timing

区域中心、边界、半径、权重与位置不得读取 lesion mask、tumor bbox、FTV、LD、SPH、BPE、HR、HER2、treatment、pCR 或 future visit。不得使用 learned attention、detector、segmentation、Grad-CAM 或 test-driven geometry。区域只以 frozen C1B crop physical center 为中心。

pCR 使用 causal prefix `T0`、`T0-T1`、`T0-T2`、`T0-T3`；T3 永久标记 late/pre-surgery。Phenotype 使用单 visit `T0/T1/T2/T3`。Clinical covariate 是 baseline-only，FTV prefix 只使用截至当前 timing 的 `log1p(FTV)`。

## 5. 预注册物理区域

区域权重使用 final feature **sampling cell** 与物理 cube 的三轴 fractional-volume overlap；不使用 nearest/binary assignment，也不把 47-voxel theoretical receptive field 当作 LOCAL sampling cell。

Primary 32/48/64 mm：

- R0：64-mm Full Local，128-D weighted mean；
- R1：32-mm Central，128-D；
- R2：48-mm cube minus 32-mm cube，128-D Inner Shell；
- R3：64-mm cube minus 48-mm cube，128-D Outer Shell；
- R4：`[R1;R2]`，256-D；
- R5：`[R1;R2;R3]`，384-D。

Outcome-blind secondary 24/40/64 mm 在所有新 label probe 前固定为 S1/S2/S3/S4/S5，定义与 primary 同构。Secondary 不参与 primary gates/classification。

R5_RP192 是低容量 control：使用 seed `260812` 生成 Gaussian `384×192` 矩阵，经 reduced QR 得到固定 orthonormal columns；每个 visit 右乘同一 `Q`。它不参与 primary gates。未经 supervised transformation 的 R5 必须始终作为 primary comparison。

Geometry QC 必须证明 `w32 + (w48-w32) + (w64-w48) = w64`、所有 shell 非负、物理体积守恒，并证明 w64 与 checkpoint buffer、C1B sidecar 和 Goal 5 LOCAL weight 三方一致。R0 pooled vector 必须与 Goal 5 raw LOCAL mean bitwise equal；R0 经 frozen checkpoint response projection 后必须与既有 LOCAL 192-D state allclose (`rtol=1e-5, atol=1e-6`)。

## 6. Probe contract

### Classification

- train-only `StandardScaler`；L2 logistic、liblinear、`max_iter=10000`；
- C grid `[1e-4,1e-3,1e-2,1e-1,1,10,100]`；validation AUROC 最大，`1e-12` tie 取更小 C；
- pCR unweighted；HR/HER2 balanced；subtype 使用 sklearn-1.8-compatible exact balanced binary-OvR，并以 validation macro-OvR AUROC 选择；
- threshold 只用 validation balanced accuracy；outer test probability 每个 model 只生成一次；不作 train+val refit。

pCR models：MRI-only `Rk`；clinical complementarity `C+Rk`；beyond-FTV `C+F+Rk`，并包含 C 与 C+F baseline。C 固定为 `C2_full_with_treatment` 既有 contract。Primary population：pCR MRI-only 同时报 full 808 与 matched 375；C/C+F analyses 使用 matched FTV-complete 375。

### FTV / delta-FTV

复用固定 Ridge：train-only X scaler；alpha `[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]`；`solver=lsqr,tol=1e-8,max_iter=10000`；validation MSE 选择，tie 取较小 alpha；test predict 一次。Static endpoint 为 `Rk(Tt)->FTV(Tt)`；delta 为 `Rk(t+1)-Rk(t) -> FTV(t+1)-FTV(t)`。Primary scope 为 measurement-valid；observable-only 为 sensitivity。

## 7. Goal 5 matched Oracle comparison

固定 Goal 5 `oracle_pair_PERI20` T0-T1 pCR population；它与 FTV-complete 375 完全相同，sorted-ID SHA-256 为 `64a7599a7903e2e013ae6ae5d50018019eee35ac408d7312f36e0c47536d29b6`。

Primary recovery denominator 按用户指定的已发布 Goal 5 contrast 冻结为 `PERI20(mean+std) - FIXED_P3(full-local mean+std)`；仅其 uplift>0 的 matched cells 计算。因此正式 Gate-C cell 是 LOCAL0/T0-T1：seed 2026 denominator `+0.03567753001715257`，seed 3026 `+0.033276157804459694`。

必须披露 representation mismatch：新 numerator 是 `Rk(raw regional means)-R0(raw full-local mean)`，而 published denominator 是 mean+std 对 mean+std。并报告两个 bridge diagnostics：

1. Goal 5 `PERI20 - P1(raw mean)`（两个 seed 并非均正，因此不作为 recovery denominator）；
2. 新 R0 predictions 与 Goal 5 P1 predictions 的精确复现，以及 projected-R0 与既有 LOCAL state 的 feature parity。

同时在相同人群列出 FIXED_P3、PERI10、PERI20、CORE、CORE_PERI 与 R1/R2/R3/R5；不跨 variant-specific population 比较绝对 AUROC。

## 8. Bootstrap、Gates 与 classification

Bootstrap 固定为 paired patient resampling within outer fold，2000 replicates，95% percentile CI，seed 260811。预注册 scope 为 early/mid timing 的 Gate candidates R2/R3/R5：MRI-only Rk-vs-R0、C+F+Rk-vs-C+F+R0，以及 LOCAL0/T0-T1 Oracle PERI20-vs-FIXED_P3。AUROC/AUPRC delta 为 comparison-reference，Brier improvement 为 reference-comparison。

- Gate A：同一 arm/timing/candidate（R2/R3/R5），timing∈T0/T0-T1/T0-T2；两个 seed `AUROC(Rk)-AUROC(R0)>0` 且两 seed 平均 ≥0.02。
- Gate B：同一 arm/timing/candidate（R1-R5），early/mid；两个 seed `AUROC(C+F+Rk)-AUROC(C+F+R0)>0`，且 `AUROC(C+F+Rk)-AUROC(C+F)` 的两-seed平均 ≥0 并非两个 seed 同为负。
- Gate C：LOCAL0/T0-T1/R1/R2/R3/R5；两个 seed numerator 均 >0，且相对各自 published positive PERI20-vs-FIXED_P3 denominator 的两-seed mean recovery ratio ≥0.30。
- Gate D：同一 arm/static visit/region R1-R5/target HR、HER2 或 subtype；相对 R0 两个 seed AUROC 均 ≥+0.03。

Classification precedence：A（Gate A+C）优先；否则若 Gate A 且 Gate B 失败为 B；否则若 Gate C 失败但存在任一双-seed同向正 mask-free gain，判 C；若 Gate A 失败且没有任何 primary candidate 双-seed同向正 gain，判 D。若组合不满足上述唯一映射，报告 `INDETERMINATE_DIAGNOSTIC` 并逐 Gate 解释，不事后改阈值。

## 9. 输出与解释边界

输出中文 `reports/final_report.md`，回答用户列出的 12 个问题；附 region schematic、occupancy statistics、MRI-only pCR、phenotype、C+F incremental、Oracle recovery、patient bootstrap、seed consistency、timing sensitivity。公开文件不得含 patient ID；private feature/prediction 只保留在 gitignored owner-only 路径。

任何 AUROC/FTV 可解码性只是冻结、内部 OOF diagnostic；不得解释成外部泛化、统计显著性、因果机制、peritumoral biology 或 molecular phenotype。粗 Z spacing、大 receptive field、complete-four-visit selection 与上游 T0 localization-centered crop 必须作为限制。

