# Four-Visit Valid-Source-Overlap Technical Eligibility Amendment + FTV-only 2×2：预注册计划

## 0. 预注册身份与不可变边界

- 新 run：`c1b_overlap_eligibility_ftv_stageb`
- 预注册时间（UTC）：`2026-08-09T11:47:43Z`
- 起始 commit：`283b4e40205df8cf47b688d350e6090eed95727a`
- 分支：`feature/c1b-overlap-eligibility-ftv-stageb`
- 此文件必须在本 run 的 cohort 统计、representation、training、FTV probe 或 Stage-B 结果产生前冻结。

本 run 是在独立 provenance audit 已判定没有 source-authoritative unique repair 后建立的全新实验，不是删除已知失败病例后追溯宣布旧 run 成功。下列旧结论和旧目录只读且不可修改：

1. `additional_experiments/c1b_model_ready_ftv_sanity/STAGE_A_NO_GO.json`：`STAGE_A_NO_GO = immutable`，预注册前 SHA-256 为 `ad2604d35c9fca645f6487c7decf297a0c8f0711136973491d537ac42aa8f080`。
2. `additional_experiments/c1b_model_ready_ftv_sanity/reports/final_report.md`：预注册前 SHA-256 为 `50d7ce177a431ae536ccddef7faf1f93dde753d9b87d0cacb4a115077b1e4976`。
3. `additional_experiments/c1b_model_ready_ftv_sanity/metrics/model_input_pipeline_h_validation_gate.json`：预注册前 SHA-256 为 `e3d4bb8d5337c1c21ec7f606b8a8a34536c67c540e7c534ba2515a7572777c8a`。
4. `additional_experiments/zero_overlap_provenance_audit/AUDIT_NOT_REPAIRABLE.json`：`AUDIT-NOT-REPAIRABLE = immutable`，预注册前 SHA-256 为 `042dd629fdb10a7b08bfdeceaa6cf51d9d9e7e713fa5a86027a4d569640d0ffd`。
5. `additional_experiments/zero_overlap_provenance_audit/reports/final_report.md`：预注册前 SHA-256 为 `2438cff32ae283cbac7a59dc547643e5411beb63278170cfc65268dc3ee8e5ed`。

历史 audit 对 `948 → 947` 的描述只作为待验证预期，不是本 run 的结果或硬编码分母。本 run 必须从完整 candidate model-input population 重新机械计算 eligibility。

## 1. 科学问题与顺序门禁

本实验只回答：更可观察的 image state 是否让 Direct FTV Grounding 在 quantitative response representation 上更忠实，并与 JEPA/base optimization 更兼容？

严格顺序为：

1. Stage A：Technical Eligibility Amendment + frozen C1B-H Model-Ready rerun。
2. 仅当所有 Stage-A gate 均通过并写出 `STAGE_A_GO.json` 后，Stage B 才获得授权。
3. Stage B：FTV-only `Legacy/C1B-H × No grounding/Direct grounding` 2×2 representation sanity。
4. 任一 Stage-A gate 失败时写出 `STAGE_A_NO_GO.json`，立即停止；不得训练、导出新 features 或运行 probe。

不得为了进入 Stage B 放宽阈值、增加 patient-specific correction、改变 C1B-H grid，或新增 exploratory preprocessing hard gate。除明确 code/data-integrity bug 外，不再搜索 preprocessing architecture。

## 2. Technical Eligibility Amendment

### 2.1 冻结规则

对完整 candidate model-input population 中的每个 patient，定义：

```text
Eligible(patient)
=
AND over t in {T0,T1,T2,T3}
[
  valid_source_voxels(patient,t) > 0
]
```

`valid_source_voxels(patient,t)` 是该 visit 的 source MRI 与冻结 C1B-H target physical grid 相交后，具有真实 source support 的 target voxels 数。只有 T0、T1、T2、T3 四个建模 visit 均大于 0 的 patient 才进入新的 longitudinal model-input population。

### 2.2 允许读取的字段

Eligibility runner 只允许读取：

- imaging source；
- raw/rebuilt DICOM geometry 与已验证的 source affine；
- 冻结 C1B-H physical grid；
- valid-source overlap。

Eligibility runner 禁止读取或间接派生：

- FTV、LD、SPH、BPE 或任何 lesion-response quantity；
- pCR 或其他 outcome；
- clinical、treatment、subtype；
- model loss、representation metric、downstream performance。

规则中不得出现 patient ID、case alias、known failure list，代码和配置不得写死 `948`、`947`、排除人数或目标 visit 数。处理流程必须是：

```text
all candidate patients
→ deterministic eligibility function
→ eligible population
```

### 2.3 输出与隐私

私有、含标识的逐例证据只允许写入：

- `manifests/technical_eligibility_patients.private.csv`
- `manifests/technical_eligibility_visits.private.csv`

公开报告/指标只能包含聚合值：candidate/eligible/excluded patients，candidate/valid/zero-overlap visits，以及 exclusion reason aggregate；不得出现 patient ID、源路径、UID、坐标或可重识别 case alias。

必须生成 `reports/technical_eligibility_amendment.md`，明确这是 outcome-free、pre-registered 的新 run，不是对旧 Stage-A NO-GO 的 post-hoc 修改。

## 3. 冻结的 C1B-H input 与复用边界

唯一正式 input strategy 继续冻结为：

```text
C1B-H = T0-Anchored Fixed Physical Detail Crop
        + Header-based longitudinal strategy
```

直接复用并记录代码 hash/version/provenance：raw-DICOM PixelData rebuild、singular-sform repair、true RAS+ canonicalization、frozen C1B-H grid、DCE7 phase contract、anti-alias + fixed-grid resampling、grounding observability mask、normalization、cache schema 和 leakage exclusion。

不得重新测试或选择 C1A、C2A、C2B、C1B-R、deformable registration、registration hyperparameter、larger FOV、different target spacing 或 adaptive recentering。不得用 image registration transform 作为 repair，不得人工 flip/translation/recenter。若发现必要 code bug，必须在独立 bug-fix ledger 中记录旧/新 hash、原因、影响和重新验证，禁止静默修改。

冻结几何与 tensor contract：

- canonical orientation：true RAS+；
- output shape ZYX：`112 × 176 × 160`；
- spacing XYZ：`0.9 × 0.9 × 2.0 mm`；
- T0-T3 共用 T0-only center/basis/shape/spacing；
- fixed-grid linear resampling，必要时 source-domain anti-alias；
- DCE7 channel semantics 与旧 frozen contract 完全一致；
- per-visit、per-channel valid-source normalization 与旧 contract 一致；
- model tensor 只能含 DCE7 image values；geometry、affine、crop metadata、valid-source mask、FTV support 和 grounding mask 均不得进入 tensor。

`grounding_observable_mask` 保持 loss-side metadata：observable visit 才贡献 Direct FTV loss；不可观察或无有效 FTV 的 visit 仍贡献 base JEPA loss。它不参与 eligibility、input construction、forward、patient filtering 或 primary probe filtering。

## 4. Stage A population、cache 与 gate

从旧 run 的完整 candidate model-input manifest 定义读取候选人，并对所有候选人重新执行四访 overlap rule；不能只运行历史失败病例。正式分母只能由本 run 的 runner 产物给出。

只对本 run 的 eligible population 构建完整 C1B-H DCE7 cache。每个 eligible patient 必须有四访并满足：finite、nonconstant、`valid_source_voxels > 0`、shape 一致、DCE7 channel semantics 一致、orientation 一致、physical grid 一致、原子写入与 cache round-trip/hash 一致。不得用 partial completion 宣布 model-ready；必须 `eligible cohort = 100% cache completion`。

Stage A GO 要求以下 15 项全部 PASS：

1. eligibility rule 在任何 Stage-B 结果前冻结；
2. eligibility 无 outcome/lesion/clinical/treatment/model leakage；
3. eligible cohort 由通用程序机械确定，无 patient-specific rule；
4. 所有 eligible visits `valid_source_voxels > 0`；
5. DICOM repair contract 持续 PASS；
6. true RAS+ orientation 持续 PASS；
7. C1B-H 仍为唯一冻结策略；
8. formal FTV support containment 持续满足原冻结阈值；
9. FTV retention Q05 `>= 0.95`；
10. grounding observability mask 仍仅为 loss-side metadata；
11. complete DCE7 cache 覆盖 100% eligible population；
12. cache round-trip/hash PASS；
13. 无 patient-specific manual correction；
14. 无 unresolved catastrophic resampling case；
15. geometry metadata 不进入 model tensor。

不得新加 exploratory preprocessing hard gate。若 15 项全部通过，写出 machine-readable `STAGE_A_GO.json` 并立即进入 Stage B；否则写出 `STAGE_A_NO_GO.json` 并停止。

## 5. Stage B 2×2 设计（仅 Stage A GO）

| Arm | Input | Objective |
|---|---|---|
| L1 | frozen legacy DCE7 | frozen JEPA/base，no FTV grounding |
| L3 | frozen legacy DCE7 | same base + Direct Static FTV Grounding |
| N1 | frozen C1B-H DCE7 | frozen JEPA/base，no FTV grounding |
| N3 | frozen C1B-H DCE7 | same base + Direct Static FTV Grounding |

L1/L3 必须在本 run 的相同 eligible cohort 上重新运行；旧 G1/G3 仅作 external historical reference。L3/N3 使用完全相同的 FTV target transform、mask、head 和 loss，固定 `lambda_FTV = 0.25`，禁止 lambda sweep。训练中禁止 ΔFTV supervision。

不修改 3-D encoder、response projection、JEPA transition、EMA、SIGReg、optimizer type、FTV head architecture 或 augmentation contract。唯一实验因素是 legacy versus C1B-H input 以及 no-grounding versus Direct FTV grounding。

### 5.1 Split、seed 与 budget

- 优先复用 locked seed-2026 patient five-fold manifest，并与本 run eligible population机械取交集；禁止重新随机分 fold或为保持人数移动 patient。
- 训练使用 seeds `2026`、`3026` × 5 folds。
- 同一 `(seed, fold)` 内四臂尽可能使用 paired initialization、paired patient order 和相同 effective sample stream。
- 四臂保持 optimizer、LR、optimizer steps、patients/epoch、max epochs、early stopping、EMA、SIGReg 与 effective batch 尽量一致。
- 目标 effective batch 为约 32。默认沿用已冻结的 physical batch 4、gradient accumulation 8；若正式 OOM，只允许四臂共同改为 physical batch 2、accumulation 16 并从头重启，不得按 arm 单独改变。
- 每个 run 必须记录 physical batch、accumulation、effective batch、selected epoch、checkpoint/hash、finite status 和初始化 hash。

### 5.2 FTV target contract

从既有 Direct Grounding 正式代码确认并复用：

```text
log(FTV + epsilon)
+ outer-fold-train-only winsorization
+ outer-fold-train-only median/IQR standardization
```

若正式旧代码的冻结细节与上述摘要不同，以已提交、经验证的正式版本为准并在 provenance 报告中逐项列出；不得猜测或使用 val/test 拟合 transform。

## 6. 冻结 endpoints 与分析

所有 probes 在 frozen encoder/response state 上运行，并严格 outer-fold isolated（仅 outer-train 拟合 scaler/Ridge）。训练 seed 是主要独立单位；5 folds 和 visits 不是独立 replicates。patient bootstrap 只描述 fitted-model conditional uncertainty。2 seeds 属 pilot，不声称最终 multiseed robustness。

### 6.1 Primary A：Static FTV

`r_t → FTV_t`，分别报告 T0/T1/T2/T3 与 macro：Spearman、Pearson、R²、RMSE、MAE、outer-train mean-baseline RMSE gain、prediction variance ratio。必须同时报告 transformed space 和 natural FTV scale，主要解释 natural-scale R²、variance compression 与 calibration。

### 6.2 Primary B：Observed ΔFTV

冻结 state 后计算 `Δr_t = r_(t+1)-r_t`，预测 literal natural-scale `ΔFTV = FTV_(t+1)-FTV_t`。分别报告 T0→T1、T1→T2、T2→T3 与 macro 的 Spearman、Pearson、R²、RMSE、B0 gain、variance ratio。训练与 checkpoint selection 禁止读取 ΔFTV。

### 6.3 Primary C：Optimization safety

比较 L3 vs L1 与 N3 vs N1 的 selected validation base degradation：`<=5% = PASS`，`>5% = FAIL`。逐 seed/fold/model 报告 selected epoch、base loss、grounding loss、total loss、base degradation、representation std、finite status 和 pass/fail。

### 6.4 三个核心比较与 DiD

1. Input effect without grounding：`N1 - L1`。
2. Grounding effect under C1B-H：`N3 - N1`。
3. Difference-in-Differences：`(N3 - N1) - (L3 - L1)`。

DiD 必须分别计算 static FTV Spearman、static natural-scale R²、ΔFTV Spearman、ΔFTV natural-scale R² 与 base degradation。不得把 Spearman 的重复小幅改善当作唯一成功；重点是 natural-scale R²、prediction variance 与 base safety。

## 7. 预冻结科学分类

### Outcome A：OBSERVABILITY BOTTLENECK SUPPORTED

若 `N3-N1` 在 natural-scale static FTV R² 和/或 ΔFTV R² 明显强于 `L3-L1`，且 N3 base safety 优于 L3，则支持 legacy partial image evidence 与 global FTV target mismatch 是 calibration/instability 的重要来源。唯一第一优先级：下一轮 FTV+LD。

### Outcome B：OBSERVABILITY IMPROVED, OPTIMIZATION BOTTLENECK REMAINS

若 N3 representation（尤其 R²）明显改善，但 base instability 与 L3 类似，则 input observability 是真实问题，而 JEPA-grounding optimization conflict 是独立瓶颈。唯一第一优先级：grounding stabilization（warm-up、two-stage 或 gradient surgery 的具体选择依据既有 conflict audit）。

### Outcome C：INPUT FIX NOT REPRESENTATION-LIMITING

若 `N1≈L1` 且 `N3-N1≈L3-L1`，static/dynamic FTV 均无明显改善，则 crop truncation 是真实 preprocessing defect，但不是当前 representation 主要限制。唯一第一优先级：stronger image encoder + richer response representation。

### Outcome D：C1B-H worse

若 N1/N3 明显差于 legacy，先完成 formal analysis，并检查 excessive padding、effective spatial resolution、larger-FOV lesion dilution 和实际 budget matching；不得立即修改 input 或启动新 input search。最终报告必须依据预冻结证据映射到唯一第一优先级。

## 8. 明确禁止项

- 本 run 禁止 FTV+LD training；即使 N3 很好，也必须先完成并停止本报告。
- pCR 不参与 Stage A、eligibility、checkpoint/crop/model/lambda selection 或主要科学分类。只有 primary analysis 完成后、成本低时才允许严格 secondary image-only frozen readout，且不得改变结论。
- 禁止用 outcome、lesion、model performance 或 registration transform 决定 technical exclusion/repair。
- 禁止继续人工 flip/translation/recenter、deformable repair、FOV/spacing sweep 或 adaptive recenter。
- 禁止将 geometry metadata 或 grounding mask放入 model tensor。

## 9. 预注册交付物

Stage A：

- Table 1：technical eligibility + Stage-A QC；
- Figure 1：technical eligibility flow；
- Figure 2：valid-source-overlap distribution；
- Figure 3：cache completion QC；
- private eligibility manifests、amendment report、gate JSON 与 provenance/hash ledger。

Stage B（仅 GO）：

- Table 2：Static FTV，四臂、T0-T3 + macro；
- Table 3：Observed ΔFTV，四臂、三 transition + macro；
- Table 4：Optimization safety；
- Table 5：Difference-in-Differences；
- Figures 4–12：Static Spearman、Static natural R²、predicted-vs-true natural FTV、ΔFTV Spearman、ΔFTV R²、base degradation heatmap、representation std、interaction plot、representative training curves。

最终生成中文 `reports/final_report.md`，逐项回答 eligibility 冻结时点、实际分母与 exclusions、100% model-ready、N1-L1、N3-N1、natural R²、observed ΔFTV、DiD、base safety、旧 G3 机制解释、当前 bottleneck，并在 FTV+LD / optimization stabilization / stronger encoder 中给出唯一第一优先级。

科学链条固定表述为：

```text
Legacy fixed voxel crop
→ poor lesion observability
→ C1B physical crop
→ ~98% available-support containment
→ full DICOM / orientation validation
→ one catastrophic longitudinal coordinate case
→ independent provenance audit
→ no authoritative repair
→ pre-registered technical eligibility amendment
→ model-ready C1B cohort
→ FTV-only causal representation sanity
```

最终问题：**Does a more observable image state make Direct Grounding more quantitatively faithful and more optimization-compatible?**
