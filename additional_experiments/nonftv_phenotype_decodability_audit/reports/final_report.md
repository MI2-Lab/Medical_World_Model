# Non-FTV Phenotype Target Decodability Audit 最终报告

## 执行结论

唯一主推荐：**`Need broader-context phenotype branch`**（优先级规则 3）。理由：Higher-priority rules failed and BPE source observability is not established in lesion-centered LOCAL FOV。本审计没有启动、也不授权自动启动 multi-target training。

Primary gates：Gate A `FAIL；固定 candidate=LOCAL3/Z2，qualifying endpoints=['T0']，minimum-seed rho macro=0.318，minimum-seed R² macro=0.060`；Gate B `FAIL；固定 candidate=LOCAL3/Z2，qualifying endpoints=无，minimum-seed rho macro=0.039，minimum-seed R² macro=0.309`；Gate C `PASS；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T0']，minimum-seed rho macro=0.143，minimum-seed R² macro=0.293`；Gate D `PASS；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T0', 'T1', 'T2']，minimum-seed rho macro=0.284，minimum-seed R² macro=0.040`；Gate E `FAIL`，qualifying targets=[]。

## 设计、证据边界与完整性

本实验冻结 C1B-H/DCE7、375 人、四访视、seed-2026 五折、两个 checkpoint seed、LOCAL0/LOCAL3、Z1–Z7 与 64-mm LOCAL support；encoder/JEPA 均未重训。20 个 frozen feature cells、3,276 个 OOF aggregate endpoints 均完成。Ridge scaler/residualizer 仅在 outer train 拟合，alpha 仅由 validation 选择，test 每 probe 只预测一次；pCR 没有被解析或用于 target、representation、timing、residualizer、alpha、gate 或推荐选择。

Goal 6 是 SHA-locked sibling evidence：384 人 primary 支持 joint `N=LD+SPH+BPE` 与 joint `N_res` 在 `Clinical+FTV` 后的 increment；它没有分别证明 LD/SPH/BPE 各自的 residual pCR increment。family-specific 证据只能写成描述性排序（LD strongest、SPH weaker、BPE weakest），不能外推为三项独立 clinical proof。当前审计则固定在 375 人 image-observability estimand。

相邻变化 `100*(x_end-x_start)/abs(x_start)` 是将 Goal 6 的 signed-percent formula 新实例化到 adjacent intervals；Goal 6 原本冻结的是 baseline-referenced change，因此不能把本审计说成复现了 Goal 6 adjacent target。动态 macro 是三个 interval endpoint metric 的无权均值；Gate E 始终逐 early interval 读取 literal `z_end-z_start`，没有用 macro、T2→T3、prefix 或 FTV+LD residual 替代。

对 SPH/BPE raw 的原 brief 没有给分类阈值。为使分类 deterministic，本报告预先统一采用明确标记的 `RAW_IMAGE_SUPPORT_FOR_CLASSIFICATION`：同一固定 deployable candidate 在 T0/T1/T2 至少两个 timing、两 seed 均 rho≥0.40 且 natural R²>0。它不是新增 primary gate，也不改变 A–E。

## 十五个问题逐项回答

### 1. FTV decodability control 如何？

Gate-A-style raw control 为 `PASS；固定 candidate=LOCAL3/Z2，qualifying endpoints=['T0', 'T1', 'T2']，minimum-seed rho macro=0.595，minimum-seed R² macro=0.129`。描述性最佳 early cell：LOCAL0/Z4，T0：两 seed rho=0.816/0.787，natural R²=0.480/0.354，n=375–375。FTV 是 response control，不是 non-FTV candidate。

### 2. LD raw 是否稳定可解码？

Gate A：`FAIL；固定 candidate=LOCAL3/Z2，qualifying endpoints=['T0']，minimum-seed rho macro=0.318，minimum-seed R² macro=0.060`。描述性最佳 early cell：LOCAL0/Z2，T0：两 seed rho=0.428/0.436，natural R²=0.119/0.177，n=375–375。

### 3. LD 去除 FTV component 后是否仍可解码？

Gate B：`FAIL；固定 candidate=LOCAL3/Z2，qualifying endpoints=无，minimum-seed rho macro=0.039，minimum-seed R² macro=0.309`。这里 rho 是 outer-fold-isolated Goal-6 transformed-standardized residual rank；R² guardrail 是 FTV conditional baseline + MRI residual readout 的 raw-target reconstruction，不是“自然单位 residual R²”。描述性最佳 cell：LOCAL3/Z2，T0：两 seed rho=0.120/0.130，reconstructed target R²=0.336/0.348，n=375–375。

### 4. SPH raw 是否可解码？

Raw classification support：`FAIL；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T0']，minimum-seed rho macro=0.270，minimum-seed R² macro=0.064`；最佳：LOCAL3/Z4，T0：两 seed rho=0.485/0.476，natural R²=0.210/0.193，n=375–375。该 support 是 scorecard operationalization，不是新增 primary gate。

### 5. SPH residual 是否可解码？

Primary FTV-residual Gate C：`PASS；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T0']，minimum-seed rho macro=0.143，minimum-seed R² macro=0.293`；最佳：LOCAL3/Z4，T0：两 seed rho=0.315/0.312，reconstructed target R²=0.370/0.371，n=375–375。FTV+LD residual 仅列在 residual matrix，未进入 Gate C 或分类。

### 6. BPE raw 与 residual 是否可解码？

Raw classification support：`FAIL；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T2']，minimum-seed rho macro=0.339，minimum-seed R² macro=0.049`；最佳：LOCAL3/Z4，T2：两 seed rho=0.459/0.410，natural R²=0.121/0.104，n=375–375。Primary FTV-residual Gate D：`PASS；固定 candidate=LOCAL3/Z4，qualifying endpoints=['T0', 'T1', 'T2']，minimum-seed rho macro=0.284，minimum-seed R² macro=0.040`；最佳：LOCAL0/Z4，T2：两 seed rho=0.390/0.354，reconstructed target R²=0.102/0.102，n=375–375。即使数值 gate 通过，也不越过下一节 FOV firewall。

### 7. BPE 是否存在 input/FOV observability 问题？

是。冻结 target 是对侧乳腺中央连续五层 fibroglandular tissue early enhancement；所有 Z1–Z7 都只来自 lesion-centered 64-mm LOCAL，Oracle 也不扩 FOV。C1B nominal tensor extent 仅可由 shape/spacing 记为 (144.0, 158.4, 224.0) mm；public lock 中没有 hash-bound BPE source ROI、laterality coordinate mapping、occupancy 或 boundary-touch audit，因此状态必须是 **`FOV_OBSERVABILITY_UNVERIFIED`**，而不是擅自称 disjoint 或 encoder failure。BPE grounding 被阻断，直到验证 ≥99% source occupancy 且无 boundary touch，或开发 broader-context branch。

### 8. 哪些 adjacent Delta target/residual 可由 literal longitudinal latent difference 解码？

Gate E 只看 primary FTV-residual 与 literal difference：

- LD: FAIL；固定 candidate=LOCAL3/Z4，qualifying endpoints=无，minimum-seed rho macro=0.039，minimum-seed R² macro=0.036; descriptive best=LOCAL3/Z3，T0->T1：两 seed rho=0.117/0.099，reconstructed target R²=0.060/0.062，n=375–375.
- SPH: FAIL；固定 candidate=LOCAL0/Z4，qualifying endpoints=无，minimum-seed rho macro=0.008，minimum-seed R² macro=0.018; descriptive best=LOCAL3/Z2，T1->T2：两 seed rho=0.048/0.067，reconstructed target R²=0.124/0.120，n=375–375.
- BPE: FAIL；固定 candidate=LOCAL0/Z4，qualifying endpoints=无，minimum-seed rho macro=0.108，minimum-seed R² macro=0.034; descriptive best=LOCAL3/Z4，T1->T2：两 seed rho=0.198/0.156，reconstructed target R²=0.048/0.040，n=375–375.

全部三个 interval 的 literal-difference 描述性最佳固定 candidate（按 minimum-over-seeds rho，再按 R²与注册顺序汇总；这不是 gate）为：

- FTV raw: T0->T1 LOCAL3/Z1 rho=0.388/0.397, R²=0.044/0.038, n=375–375；T1->T2 LOCAL0/Z4 rho=0.290/0.307, R²=0.014/0.013, n=375–375；T2->T3 LOCAL3/Z3 rho=0.229/0.185, R²=-0.006/0.005, n=375–375。
- LD raw: T0->T1 LOCAL3/Z4 rho=0.117/0.109, R²=-0.011/0.004, n=375–375；T1->T2 LOCAL0/Z2 rho=-0.003/-0.012, R²=-0.018/-0.017, n=370–370；T2->T3 LOCAL0/Z4 rho=0.020/0.094, R²=-0.012/-0.034, n=313–313。
- SPH raw: T0->T1 LOCAL0/Z3 rho=-0.010/-0.021, R²=-0.073/-0.049, n=375–375；T1->T2 LOCAL0/Z4 rho=0.069/0.110, R²=-0.021/0.025, n=375–375；T2->T3 LOCAL3/Z1 rho=0.137/0.142, R²=0.029/0.006, n=375–375。
- BPE raw: T0->T1 LOCAL3/Z4 rho=0.222/0.136, R²=0.034/0.016, n=375–375；T1->T2 LOCAL3/Z3 rho=0.205/0.180, R²=0.006/0.025, n=375–375；T2->T3 LOCAL3/Z4 rho=0.134/0.144, R²=-0.002/0.017, n=375–375。
- LD FTV-residual: T0->T1 LOCAL3/Z3 rho=0.117/0.099, R²=0.060/0.062, n=375–375；T1->T2 LOCAL0/Z4 rho=0.013/0.065, R²=0.021/0.036, n=370–370；T2->T3 LOCAL3/Z2 rho=0.021/0.047, R²=0.012/-0.035, n=313–313。
- SPH FTV-residual: T0->T1 LOCAL0/Z4 rho=0.049/0.028, R²=-0.041/-0.056, n=375–375；T1->T2 LOCAL3/Z2 rho=0.048/0.067, R²=0.124/0.120, n=375–375；T2->T3 LOCAL3/Z1 rho=0.120/0.101, R²=0.124/0.106, n=375–375。
- BPE FTV-residual: T0->T1 LOCAL0/Z4 rho=0.117/0.069, R²=0.099/0.066, n=375–375；T1->T2 LOCAL3/Z4 rho=0.198/0.156, R²=0.048/0.040, n=375–375；T2->T3 LOCAL0/Z4 rho=0.152/0.176, R²=0.019/0.024, n=375–375。

Prefix sensitivity 与 dynamic macro 仍完整公开，但不授权 Gate E。

### 9. Z2 `r` 是否优于 Z1 `projector(r)`？

按同 arm/target/endpoint、两 seed 各自 `Z2-Z1 >= +0.10` 判断：LD 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.052。 SPH 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.029。 BPE 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.064。 Raw-only effect 在诊断表中另列，不能冒充 beyond-FTV evidence。

同一 representation-location audit 的 pooling 补充结果（两 seed `Z4-Z3 >= +0.10`）为：LD 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.043。 SPH 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.039。 BPE 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.083。

### 10. Oracle CORE/PERI10/PERI20 是否显著改善 target，并基于何种 validity cohort？

这里“显著改善”严格指预注册 effect threshold，不是 p-value：同一 matched eligible set 上 Oracle mean+SD 相对 matched Z4、两 seed 都 Δrho≥+0.10。LD 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.039。 SPH 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.081。 BPE 未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho=0.027。

Validity（static T0/T1/T2/T3；dynamic T0→T1/T1→T2/T2→T3）：

- CORE (Z5): static [375, 375, 374, 374]；adjacent pairs [375, 374, 373]。
- PERI10 (Z6): static [375, 375, 374, 375]；adjacent pairs [375, 374, 374]。
- PERI20 (Z7): static [375, 375, 375, 375]；adjacent pairs [375, 375, 375]。

Oracle 是 mask-dependent diagnostic，不能成为 deployment input。

### 11. 每个 target 的主要 bottleneck 是什么？

- LD：`encoder/current_feature_map`；classification=`Class D — CURRENTLY NOT IMAGE-OBSERVABLE`。
- SPH：`mixed_or_unresolved`；classification=`MIXED OR UNRESOLVED`。
- BPE：`input_observability`；classification=`FOV-BLOCKED — A/D CLASSIFICATION NOT AUTHORIZED`。

只有在 Z1–Z7 绝对 residual signal 均弱且无 projection/pooling/localization effect 时才使用 encoder/current-feature-map；其余组合保留 mixed/unresolved。BPE FOV 未验证时永远先归 input observability。

### 12. 哪些 target 值得进入下一轮 grounding？

| target | classification | raw support | beyond FTV | early dynamic | input observability | bottleneck |
|---|---|---:|---:|---:|---|---|
| LD | Class D — CURRENTLY NOT IMAGE-OBSERVABLE | False | False | False | LESION_DERIVED_TARGET_COMPATIBLE_WITH_LOCAL_INPUT | encoder/current_feature_map |
| SPH | MIXED OR UNRESOLVED | False | True | False | LESION_DERIVED_TARGET_COMPATIBLE_WITH_LOCAL_INPUT | mixed_or_unresolved |
| BPE | FOV-BLOCKED — A/D CLASSIFICATION NOT AUTHORIZED | False | True | False | FOV_OBSERVABILITY_UNVERIFIED | input_observability |

Scorecard 没有计算 weighted total。Class A 仍只是下一阶段设计候选，不等于已单 family 证明 pCR increment，也不自动启动训练。

### 13. `FTV+LD` 是否得到充分的 image-observability/beyond-FTV 支持？

结论由 LD classification 决定：**`Class D — CURRENTLY NOT IMAGE-OBSERVABLE`**。只有 Class A 才称 `FTV+LD` 得到充分支持；否则 LD 只可称 morphology/extent response target。最终 recommendation=`Need broader-context phenotype branch`。

### 14. SPH/BPE 是否应等待 region-aware 或 broader-context architecture？

SPH classification=`MIXED OR UNRESOLVED`；若 residual Oracle localization 通过，应先开发 mask-free region-aware representation。BPE status=`FOV_OBSERVABILITY_UNVERIFIED`，故无论 probe 数值如何，都应等待可覆盖/验证对侧 tissue 的 broader-context pathway，不应在现有 LOCAL 中直接 grounding。

### 15. 下一轮正式 training 必须继续冻结什么？

必须继续冻结：C1B-H/DCE7 与 exact 375-person eligible population；canonical six-digit exact join；seed-2026 outer folds；FTV/LD/SPH/BPE workbook fields 与 hashes；T0–T3 timing（T3 始终 late/pre-surgery）；adjacent formula/zero-denominator fail-closed rule；train-only 1/99% winsor、family transform、scaler 与 fixed-alpha residualizer；selected LOCAL0/LOCAL3 checkpoints、online pathway、64-mm physical support 与 representation definitions；validation-only alpha selection/test-once contract；pCR firewall；patient-level artifact private/ignored/0600 contract。任何新 target、FOV、region-aware state 或 multi-target loss都需另立 preregistration，不能由本审计自动触发。

## Scorecard 与推荐

| target | classification | raw support | beyond FTV | early dynamic | input observability | bottleneck |
|---|---|---:|---:|---:|---|---|
| LD | Class D — CURRENTLY NOT IMAGE-OBSERVABLE | False | False | False | LESION_DERIVED_TARGET_COMPATIBLE_WITH_LOCAL_INPUT | encoder/current_feature_map |
| SPH | MIXED OR UNRESOLVED | False | True | False | LESION_DERIVED_TARGET_COMPATIBLE_WITH_LOCAL_INPUT | mixed_or_unresolved |
| BPE | FOV-BLOCKED — A/D CLASSIFICATION NOT AUTHORIZED | False | True | False | FOV_OBSERVABILITY_UNVERIFIED | input_observability |

唯一 recommendation：`Need broader-context phenotype branch`。没有 weighted total，没有用 test pCR 作 tie-break。

## Reproducibility、privacy 与交付状态

- Branch：`feature/nonftv-phenotype-decodability-audit`
- Experiment parent：`7742d737d92ed153b5c721cd323528b0a127d5ef`
- Preregistration lock SHA-256：`b3e9809f47a13b2db2c958cee4bec112b18273de75606c05538bc2fc04f706ee`；verification=`PASS`；binding count=133。
- Reported experiment commit SHA：`67c3355bded6cc79098924b9bf5bb99f4819a3ed`
- Push status：`PUSHED`
- Formal run status：`COMPLETE`；encoder retrained=`False`；pCR read=`False`；test used for alpha=`False`。
- Patient set SHA-256：`64a7599a7903e2e013ae6ae5d50018019eee35ac408d7312f36e0c47536d29b6`；公开表不含 patient identifiers，private OOF renderer 从未打开。

本报告生成时公开 aggregate artifact hashes：

- `figures/bottleneck_maximum_two_seed_gain.png` — 5eab5685da41d509a49083d4f7d5a0a8be9efb313e779d01ecfda8574165c6ed (104505 bytes)
- `figures/dynamic_ftv_residual_difference_decodability.png` — aca1707862a7208710a5e6e1124646ef380dedf1b8f887c8dab6b389bf8f96ec (187420 bytes)
- `figures/static_ftv_residual_decodability.png` — 56dff484fcbe4f600e430f02515c8dbd0df8c79208ed33ec8796dc9f4363f430 (196682 bytes)
- `figures/static_raw_decodability.png` — 7c711066263d3919172bcb97c87a31c5db1ccacceec439b9694c86a0dcb82c99 (233615 bytes)
- `metrics/bottleneck_diagnostics.csv` — 129fcd8daee76d44aff954ff98427b3f29227df320304e19a528688fdc0704dc (142298 bytes)
- `metrics/bpe_fov_observability_audit.csv` — d8c2cf5815174da5630432de05f89e7a87f1433547a3d1d6dd3cec8575d93656 (1406 bytes)
- `metrics/descriptive_best_cells.csv` — b9f3f4d72cdd8cdc1489d4584b0f4c956f5014bdd7b73ef5b1333094dead2e97 (5521 bytes)
- `metrics/dynamic_interval_best_cells.csv` — c0a8980d122c0d5b182f40eb4c6008eca539679aa758ec4398b94e23732d87e9 (8820 bytes)
- `metrics/dynamic_macro.csv` — a863bb4d364e169b0b9f644c2202239933700be9182dd248bf7e12732fe1eff5 (623452 bytes)
- `metrics/final_target_recommendation.csv` — f9ee6df833b16b642deeb8a2af5bfb733fb67b804b03cdcdf668e963fee939c0 (611 bytes)
- `metrics/gate_candidate_matrix.csv` — f3d6d238fc4e40a192ccb0f7b7ca594e39d86178696decf5ea134c8076c284ba (110753 bytes)
- `metrics/grounding_candidate_scorecard.csv` — 45cce63c01c26ffd4977628672114b256b9963a0f5cf2cbc800d26f623d7fb47 (2192 bytes)
- `metrics/oracle_validity_summary.csv` — 5182dc89350306605fc2efa2dc74d4e48b4e42e32af7d17b42276092edc4c36f (2936 bytes)
- `metrics/primary_gates.json` — 11907a98261048a0629ce2ff041ad3b6e95c5d8b3b489808d3a1c3d663084811 (28688 bytes)
- `metrics/probe_integrity_summary.csv` — 5f029c29940fbe39d83a5b08917d480239b036415f1f43b2334a42bac786feb0 (15081 bytes)
- `manifests/public_analysis_artifacts.csv` — f8b7d9761ece560d7d022d989084ef3dbb4e86fc6edd42654e3c67f20940d1d4 (1781 bytes)

Commit/push 字段通过命令行注入，便于在 scientific commit 后记录 delivery provenance；若 push 失败必须保留 local commit、写 `GITHUB_PUSH_FAILED` 与原始错误，禁止 force push。

## 冻结后报告呈现勘误（presentation-only）

冻结分析器的 `_metric_sentence` 在计算 R² 标签后遗漏返回语句，导致 10 个描述性最佳摘要未显示。该问题在全部正式聚合结果与验证完成后才被发现。交付工具 `errata/apply_report_presentation_erratum.py`（SHA-256 `1be164a6bbcc5126f723c0b8577d7e2d399c56e2f68b0ed2ae678829ebecd714`）先完整验证原锁、133 个绑定与冻结分析器哈希，再仅在内存中恢复原本已经写在相邻不可达代码块中的返回格式，并调用冻结分析器的 `--report-only` 路径。

这是锁定后、未纳入预注册的纯呈现勘误：没有重跑或重拟合任何模型，没有修改target、cohort、fold、representation、threshold、metric、gate、classification、scorecard、figure 或 recommendation；private OOF 只做压缩文件字节哈希校验，从未被本工具打开解析。完整前后哈希与交付链记录在 `errata/` 的只读 JSON manifest。
