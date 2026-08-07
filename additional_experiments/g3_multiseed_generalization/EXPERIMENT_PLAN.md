# G3 Multi-seed Generalization 冻结实验计划

本计划在任何正式训练开始前冻结。正式训练后不得删除表现差的 seed/fold、修改阈值、重选 `lambda_FTV`，也不得以 exploratory 结果替代正式结论。

## 1. 科学问题与边界

本实验只检验：严格 DCE7、GAP pooling、Direct static FTV grounding 的 G3 正向结果，能否跨独立训练随机种子和固定 outer folds 重复，以及上一轮 fold 3 的 validation base-loss failure 是偶然波动、fold-specific 冲突还是一般优化不稳定。

本轮只重新训练 G1/G3；不运行 G0/G2/G4，不修改 encoder、response projection、projector、transition、loss 形式或 checkpoint selection。禁止加入 LD、sphericity、BPE、ΔFTV、clinical、treatment、pCR encoder supervision、direction/magnitude/path loss或 pretrained encoder。

## 2. 起始 provenance 与不可修改资产

- repository branch：`feature/ispy-clean-corejepa`
- 起始 HEAD：`596d6d509aaf62c5385344a83b8ed66dd301ee79`
- 上一轮公开实验：`additional_experiments/direct_grounded_response_state/`
- 上一轮训练实现 SHA-256：`fb308f8a3cfe735ca1ef2e17e66367b11d4e6edc424bd01725cad200b780e750`
- 上一轮公开源码训练实现 SHA-256：`d2ee8a74c9fa68f78f553953e6624d3fe3b5a6bc2a30b8183d6b8f869a492d3e`
- 上一轮训练计划 SHA-256：`fd43c11d9855c62d97dcd89f3ab4c46c6292be31faf7c66fa92ff1477c24dfd9`

只读引用但绝不修改或覆盖：

- `direct_grounded_response_state/`
- `observed_state_radiomics_audit/`
- `radiomics_next_change/`
- 根目录已有未跟踪 `shortcut_audit/`
- fold manifest、既有 cache、旧 checkpoints/features/predictions

新实验的 checkpoints、features、predictions、metrics 和 logs 只能写入 `additional_experiments/g3_multiseed_generalization/`。

## 3. 环境与数据锁

- conda environment：`bowen`
- Python：3.11.14
- PyTorch：2.9.1+cu130
- CUDA runtime：13.0
- GPU：3 × NVIDIA RTX PRO 6000 Blackwell Max-Q，单卡约 96 GiB
- 数据根以 `${DGRS_DATA_ROOT}` 注入，不在公开文件保存本机绝对路径。

固定 fold manifest：

`${DGRS_DATA_ROOT}/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv`

manifest SHA-256：`143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。

所有训练 seed 使用完全相同的 outer five-fold patient split。fold 0–2 的 I-SPY2 train/val/test 为 525/121/162，fold 3–4 为 526/121/161；每名 I-SPY2 患者恰好一次作为 test。808 名 I-SPY2 均有四访 pCR label；375 名有四访 FTV；另有 156 名 I-SPY1 只贡献 base pretraining loss。

## 4. 模型定义锁

G1：

```text
DCE7 -> 3-D encoder -> GAP -> response state r
     -> projector -> M0-style causal JEPA / Next-State objective
```

G3：

```text
DCE7 -> 3-D encoder -> GAP -> response state r
     ├-> projector -> M0-style causal JEPA / Next-State objective
     └-> Linear(192,1) -> static FTV auxiliary objective
```

两者都严格拒绝 binary mask 进入 backbone 或 pooling；observed response state 固定为 192-D online pre-projector `r`。FTV 只在训练时进入 auxiliary target，绝不作为 model forward input；FTV head 不用于冻结特征、Ridge probe 或 pCR readout。

G3 objective 固定为：

`L = L_base + 0.25 * L_FTV`

锁定 `lambda_FTV=0.25`，禁止任何新 lambda pilot/sweep。

## 5. 训练随机种子锁

五个预注册 `seed_base`：

`{2026, 3026, 4026, 5026, 6026}`

严格复用上一轮定义：每个 run 的 `effective_seed = seed_base + fold`。每个资产必须同时保存 `seed_base`、fold 与 effective seed。训练后不得排除、替换或新增 seed。

每个 `seed_base × fold` 必须先训练配对 G1，再以该 G1 checkpoint 训练 G3。G1/G3 必须满足：

- common-module initialization SHA 完全相同；
- train/val/test/pretrain patient hashes 完全相同；
- DataLoader patient-order policy 和 generator seed 完全相同；
- optimizer、loss 公共部分、训练预算完全相同；
- FTV transform 完全相同；
- G3 baseline checkpoint 必须来自同一 `seed_base × fold` 的 G1。

核心统计永远是 `G3(seed_base,fold) − G1(seed_base,fold)`；禁止把新 G3 与旧固定 G1 做正式比较。

## 6. 训练与 checkpoint selection 锁

统一配置：batch size 32、workers 4、最多 12 epochs、patience 4、AdamW learning rate `5e-5`、weight decay `1e-4`、EMA momentum 0.996、max grad norm 5.0、dropout 0.1、SIGReg weight 0.09/256 projections、step weights `[2,1,0.5]`、`deterministic_algorithms=false`。

训练 loader 保持上一轮 `shuffle=true, drop_last=true`：681/682 名 pretrain-train patients 每 epoch 实际处理 672 名；这是冻结 protocol，不在本轮修复。

FTV transform 每 fold 只在 outer-train measurement patients 上拟合：`log(FTV+epsilon)`、1%/99% winsorization、median/IQR 标准化；validation/test 只应用，不拟合。

G1 checkpoint：在 finite、non-collapse epochs 中选择 validation normalized next-state `state_loss` 最低者。该 `state_loss` 不含 SIGReg。

G3 checkpoint：先要求 finite、representation std≥0.05 且 `val_state_loss_G3 ≤ 1.05 × paired val_state_loss_G1`，再选择 validation FTV loss 最低者。如无 epoch 通过 base gate，保留上一轮定义的 fallback：先最小化 base-gate violation，再最小化 validation FTV loss；该 run 必须计为 base failure，不能删除。

所有 checkpoint selection 与 lambda 均不得使用 test。

## 7. 输出命名与覆盖保护

正式路径固定为：

```text
checkpoints/formal/seed_<seed_base>/g1/fold_<fold>/
checkpoints/formal/seed_<seed_base>/g3/fold_<fold>/
metrics/training/formal/seed_<seed_base>/g1/fold_<fold>.csv
metrics/training/formal/seed_<seed_base>/g3/fold_<fold>.csv
features/seed_<seed_base>/G1/fold_<fold>/
features/seed_<seed_base>/G3/fold_<fold>/
predictions/representation_probes/seed_<seed_base>/G1/fold_<fold>/
predictions/representation_probes/seed_<seed_base>/G3/fold_<fold>/
predictions/pcr_readouts/seed_<seed_base>/G1/fold_<fold>/
predictions/pcr_readouts/seed_<seed_base>/G3/fold_<fold>/
```

正式 writer 默认拒绝覆盖。任何 restart 只能跳过经 hash/schema 完整验证的 finalized asset；不得静默覆盖 incomplete 或 mismatched run。

## 8. Multi-seed smoke gate

正式训练前在真实 cache 上至少运行两个 seed bases（2026、3026）× fold 3 × G1/G3 的一 epoch smoke。必须验证：

- 两个 seed 内 G1=G3 shared initialization；不同 seed 的 initialization 不同；
- effective seed 分别等于 `seed_base+3`；
- G1/G3 split、patient-order policy、transform 和公共配置相同；
- G3 lambda 恰为 0.25，FTV gradients 到 encoder/response projection/head 均非零；
- G1/G3 architecture、mask rejection、FTV inference prohibition 与上一轮一致；
- loss/std finite，feature/probe/pCR synthetic self-tests 通过；
- smoke 不读取 test 做选择且不能污染 formal paths。

任一项失败则不得开始正式训练。

## 9. Optimization robustness endpoint

对 50 个 `seed_base × fold × model` 保存 selected epoch/mode、validation state/base/FTV/total loss、representation std、finite status、checkpoint/selection/history/initialization/split hashes。

每个 G3 定义：

`D[s,f] = (val_state_loss_G3(selected) − val_state_loss_G1(selected)) / max(val_state_loss_G1(selected), 1e-12)`

base pass 当且仅当 `D[s,f]` finite 且 `D[s,f] ≤ 0.05 + 1e-12`。

fold 3 单独报告每个 seed 的 G1/G3 validation state loss 与 degradation。上一轮 `+9.5934%` 只作外部 reference，不进入本轮 pass rate 或 CI。

fold 3 failure 解释锁定为：

- 0/5：不复现；
- 1/5：孤立/seed-dependent；
- 2/5：少数重复、提示冲突但不足以称系统；
- ≥3/5：fold 3 systematic conflict。

若 failure 分散到至少两个 folds 且无 fold 达 3/5，则解释为 general seed×fold optimization instability。

## 10. Frozen static FTV probe

每个 finalized checkpoint 仅提取一次 `[808,4,192]` frozen `r`，共 50 个 feature assets；必须 finite、canonical patient order/split/label closure 完整。禁止调用 FTV head。

严格复用上一轮 outer-train StandardScaler/Ridge、validation-only alpha selection、test predict-once protocol。alpha grid：`{1e-4,1e-3,1e-2,1e-1,1,10,100,1000}`，solver `lsqr`，tol `1e-8`，max_iter 10000。

只评估 FTV：`r_t -> FTV_t`，T0–T3。报告 Spearman、Pearson、R²、MAE、RMSE、B0 RMSE gain、prediction/target variance ratio。训练 FTV head 不得用于评估。

每个 seed 的 primary static effect：

`dS[s] = (1/4) Σ_t [rho_OOF(G3,s,t) − rho_OOF(G1,s,t)]`

每个 rho 必须在该 seed 五折合并的 375 名唯一 OOF patients 上计算；禁止用五个 fold rho 的平均替代。

## 11. Frozen observed ΔFTV probe

训练中继续禁止 ΔFTV loss。构造 `Δr=r_(t+1)−r_t`，评估 T0→T1、T1→T2、T2→T3 的 observed ΔFTV。

报告 Spearman、Pearson、R²、MAE、RMSE、B0 RMSE gain、variance ratio。

每个 seed 的 primary dynamic effect：

`dD[s] = (1/3) Σ_q [rho_OOF(G3,s,q) − rho_OOF(G1,s,q)]`

同样使用该 seed 合并五折后的 375 名唯一 OOF patients。

另计算 `dS[s,f]`（四时点 within-fold test Δrho 等权平均）和 `dD[s,f]`（三 transitions within-fold test Δrho 等权平均），只用于 fold diagnostics/heatmap/variance decomposition，不能替代 seed-level OOF endpoint。

## 12. pCR secondary endpoint

pCR 不进入 R1–R4 或三级决策；机器字段固定 `pcr_used_in_decision=false`。

严格 image-state-only readout：

- T0：`r0`（192-D）
- T0–T1：`[r0,r1,r1-r0]`（576-D）
- T0–T2：`[r0,r1,r2,r1-r0,r2-r1,r2-r0]`（1152-D）

禁止 clinical、treatment、FTV、radiomics、geometry、mask feature 与 FTV-head output。每个 training seed/fold 独立执行以下冻结协议：StandardScaler 只在 outer-train 拟合；class-balanced LogisticRegression；penalty `{l1,l2}`；C `{0.001,0.003,0.01,0.03,0.1,0.3,1,3,10}`；solver `liblinear`、max_iter 20000、tol `1e-7`。validation model selection 依次最大化 AUROC、最大化 AUPRC、选择较小 C、再按 l1→l2；validation Youden threshold 并列时先选最接近 0.5，再选较小 threshold。锁定后 test 只 predict_proba 一次。readout RNG seed 对所有 representation seeds 固定为 2026（内部仍按 fold/decision point 确定性派生），避免引入额外 readout-seed 变化。

报告 AUROC 与 sklearn average precision（表中称 AUPRC），并描述 accuracy/sensitivity/specificity。定义 `dP_AUROC[s,dp]=AUROC_G3−AUROC_G1`，AUPRC 同理；seed-level longitudinal macro 是 T0–T1 与 T0–T2 两个 paired 差值的等权平均，T0 单列。每个 seed 做 2,000 次 exact-patient paired bootstrap，同一 patient draw 跨 G1/G3 与 decision points 复用；跨 seed 报 mean、sample SD、min/median/max 与 t-CI。pCR 全部 uncertainty 只作 secondary evidence，不进入 gate。

## 13. Primary seed uncertainty 与 paired bootstrap

对 `dS[s]`、`dD[s]` 报告 across-seed mean、sample SD（ddof=1）、min/median/max、positive n/rate，以及：

`mean ± t_(0.975,S-1) * SD / sqrt(S)`

该 95% seed-level t-CI 是 R1/R2 的正式 uncertainty gate。exact sign test/Clopper–Pearson 只作描述，不增加 gate。

每个 seed 另做 2,000 次 patient-within-outer-fold paired bootstrap：同一 draw 必须跨 G1/G3 与全部 macro cells 复用，输出 seed-conditional CI。

支持性 crossed bootstrap 做 5,000 次：每次重采 training seeds，并在每个 outer fold 同步重采 patients；同一 patient draw 跨被抽中的所有 seed、G1/G3 与 cells 复用。该 percentile CI 只作敏感性分析，正式决策仍使用 seed t-CI。

bootstrap RNG seed 固定为 20260807。

另定义目标要求的动态 R² 描述量：

`dD_R2[s] = (1/3) Σ_q [R²_OOF(G3,s,q) − R²_OOF(G1,s,q)]`

其 OOF pooling 与 `dD[s]` 完全相同，但不进入 R2 gate。

## 14. R1–R4 机械判定

所有 gate 使用未四舍五入值。

R1 Static reproducibility 当且仅当：

1. 五个 `dS[s] > 0`；
2. `mean(dS) ≥ 0.05`；
3. seed-level 95% t-CI lower > 0；
4. 每个 leave-one-seed-out mean > 0；
5. 删除任一 outer fold、重新从剩余 OOF patients 计算各 seed effect 后，其 across-seed mean > 0。

R2 Dynamic reproducibility 当且仅当：

1. 至少 4/5 个 `dD[s] > 0`；
2. `mean(dD) ≥ 0.05`；
3. seed-level 95% t-CI lower > 0；
4. 每个 leave-one-seed-out mean > 0；
5. 删除任一 outer fold后重新计算的 across-seed mean > 0。

R3 Optimization safety 当且仅当：

1. 至少 23/25 个 G3 seed×fold base pass；
2. 每个 fold 的 base failures ≤2/5（不存在多数 seed 的 fold-specific failure）。

R4 No collapse 当且仅当：全部 50 个 G1/G3 selected runs 的 required loss scalars/checkpoint tensors finite、validation representation std≥0.05，且全部 50 个 frozen feature arrays finite。

## 15. No-single-seed/fold drive

R1/R2 中的所有 leave-one-seed-out 与 leave-one-fold-out 条件共同定义 `no_single_seed_or_fold_drive`。Spearman 非线性且 folds 大小不同，leave-one-fold-out 必须从患者级 predictions 删除该 fold 后重新计算 rho；不得从预先算好的 fold rho 做代数平均。

## 16. Two-way variance decomposition

primary grid 为 `dD[s,f]` 的 5×5 balanced matrix。令 grand mean、seed row mean、fold column mean 分别为 `g,r_s,c_f`：

- `SS_seed = 5 * Σ_s (r_s-g)^2`
- `SS_fold = 5 * Σ_f (c_f-g)^2`
- `SS_res = Σ_sf (dD[s,f]-r_s-c_f+g)^2`
- df 分别为 4、4、16；`MS=SS/df`
- raw seed variance=`(MS_seed-MS_res)/5`
- raw fold variance=`(MS_fold-MS_res)/5`
- residual/interaction variance=`MS_res`

表中同时保留 raw component（允许负）、clipped component=`max(raw,0)`、`sqrt_component=sqrt(clipped_component)` 与 clipped share。若三个 clipped component 总和为 0，则三个 share 全置 0 且 `dominance=no_variation`；否则只有某一 clipped share >0.5 才称该来源主导，均不超过 0.5 则结论为混合。无 cell replication，因此 residual 必须称为“seed×fold interaction + metric sampling error”。Static effect 与 base degradation可作 secondary decomposition，不替代 dynamic primary。

## 17. 三级正式结论

按下列优先级机械判定：

- `ROBUST` 当且仅当 R1∧R2∧R3∧R4。
- `PROMISING BUT UNSTABLE` 当且仅当 R1∧R2、但不满足 ROBUST，同时 `no_single_seed_or_fold_drive=true` 且没有任何 fold 达多数 base failure。所有失败仍完整保留，解释为 seed/seed×fold optimization instability。
- `NOT ROBUST`：其余全部情况。R1/R2 任一失败、leave-one-out 变号、某 fold 多数 seed base failure，或任何 endpoint 因 nonfinite/缺 seed 不可验证，均落入该类。

pCR 不得改变此结论。

## 18. 必须输出的机器表

至少输出：

1. `training_stability_seed_fold.csv`
2. `probe_seed_cell_metrics.csv`
3. `probe_seed_fold_cell_metrics.csv`
4. `seed_fold_effects.csv`
5. `seed_level_robustness.csv`
6. `fold_level_robustness.csv`
7. `seed_uncertainty.csv`
8. `conditional_seed_bootstrap_ci.csv`
9. `crossed_bootstrap_ci.csv`
10. `leave_one_out_sensitivity.csv`
11. `variance_decomposition.csv`
12. `pcr_secondary_seed_metrics.csv`
13. `decision_gates.csv`
14. `decision.json`
15. input/history/selection/prediction/figure manifests 与 coverage/issues 表。

其中最小 schema 锁定为：

- `seed_level_robustness.csv` 必含 `seed_base`、static Δρ、ΔFTV Δρ、ΔFTV ΔR²、pCR longitudinal ΔAUROC、failed-fold count；
- `fold_level_robustness.csv` 必含 `fold`、static Δρ mean/SD、ΔFTV Δρ mean/SD、base failure count/rate；
- `seed_fold_effects.csv` 必含 `seed_base`、`fold`、`dS_sf`、`dD_sf`、`D`、`base_pass`，base degradation matrix 必须从该表生成。

所有公开表必须聚合且不含 patient-level rows。

## 19. 必须生成的图

至少生成并解码验证：

1. 每 seed static FTV G3−G1 ΔSpearman 与 conditional patient CI；
2. 每 seed observed ΔFTV G3−G1 ΔSpearman 与 CI；
3. seed×fold base degradation heatmap；
4. seed×fold ΔFTV improvement heatmap；
5. fold 3 across-seed base degradation（含旧 +9.5934% 外部参考）；
6. static gain distribution；
7. ΔFTV gain distribution；
8. pCR AUROC secondary paired comparison；
9. seed/fold/interaction variance summary；
10. fold-level static/dynamic mean±SD；
11. selected epoch 与 representation std safety panel。

## 20. 验收硬门槛

- seeds 与本计划完全一致，只含 G1/G3；
- 恰有 50 finalized best checkpoints、50 histories、50 selections，model×seed×fold 唯一；
- 每格 G1=G3 shared-init、split、transform、公共 config；不同 seed initialization 非全相同；
- G3 lambda 全为 0.25且从未 sweep；architecture/transition 与上一轮 G1/G3 contract 相同；
- 50 histories required columns/唯一 selected epoch完整，test flags全false；
- 50 features 均 `[808,4,192]`、finite、canonical closure/source checkpoint hash闭合；
- 50 FTV-only probe files与50 pCR files覆盖唯一键、test-once guard、train/validation-only selection；
- 预期 test prediction rows：probe 26,250；pCR 24,240；每 seed分别覆盖375/808唯一 patients；
- aggregated metrics 能从 patient-level predictions 独立重算一致；
- variance decomposition 先通过 constant/seed-only/fold-only synthetic tests；
- decision 可从未舍入表独立机械重算，且 `pcr_used_in_decision=false`；
- 至少 11 PNG，manifest/path/hash/bytes/decode 一一闭合；
- 中文报告明确回答目标中的七个问题并引用机器证据；
- checkpoint/feature/patient predictions/logs/非 final metrics 不进入 Git；公开文件无患者 ID、真实本机绝对路径或 secrets。

## 21. 最终报告必须回答

1. G3 static FTV improvement 是否跨 seed 可重复？
2. G3 observed ΔFTV improvement 是否跨 seed 可重复？
3. 前一轮 fold 3 failure 是否重复？
4. instability 主要来自 seed、fold，还是混合/interaction？
5. G3 属于 ROBUST、PROMISING BUT UNSTABLE 或 NOT ROBUST？
6. 是否值得把 G3 作为 Factorized Grounded Response State 基础？
7. 下一步应扩展 grounding targets，还是先解决 optimization？

## 22. Optional exploratory 明确隔离

只有在全部正式结果、统计和三级结论冻结后，才能探索 EMA/last-k checkpoint averaging；必须写入独立 `exploratory/` namespace，不能替换任何正式 checkpoint、prediction、metric 或结论。本轮默认不执行该 optional 分析。

## 23. 执行汇报与资源预算

所有用户可见进度、Markdown 报告与图表文字均使用中文；最终主报告固定为 `reports/final_report.md`。必须在以下七个节点主动汇报：protocol/资产复核、计划冻结、smoke gate、50-run 训练完成、feature/probe/pCR 完成、统计与三级结论完成、隐私验收及 GitHub push 完成。

预算依据上一轮实测：单次训练约 84–106 秒、峰值显存约 28.6 GiB；5 seeds 正式 checkpoints 约 3.66 GiB、features 约 136 MiB、predictions 约 54 MiB。三张约 96 GiB GPU 各自串行执行 pair，完整训练保守预算 35–50 分钟。资源预算只影响调度，不允许减少预注册 seeds/folds/models。
