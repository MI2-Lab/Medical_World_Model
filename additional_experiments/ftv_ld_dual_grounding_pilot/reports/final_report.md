# FTV + LD Dual-Grounding Pilot 最终报告

## 1. 科学问题

本实验先问：当前 DCE7 lesion-centered fixed crop 是否完整观察到 longest diameter（LD）所需的病灶空间范围？只有该 outcome-free observability gate 通过，才允许问第二个问题：`JEPA + FTV + LD` 是否比 `JEPA + FTV` 学到更丰富的 image-derived longitudinal response state。

执行分支为 `feature/ftv-ld-dual-grounding-pilot`，source commit 为 `91ce7e5a26ef3674c56e56e00fe2efa76fdb841b`。运行环境为 conda `bowen`、Python 3.11.14、PyTorch 2.9.1+cu130、CUDA runtime 13.0；可见硬件为 3 张 NVIDIA RTX PRO 6000 Blackwell Max-Q（每张约 96 GiB）。Stage A 没有读取 pCR，也没有训练模型。

## 2. 为什么选择 LD

LD 是既有 screening 的 conditional/pragmatic first candidate，不是统计上唯一优胜者。它比 SPH 少一个 HIGH mask-geometry shortcut gate，也没有 BPE 的对侧乳腺 input mismatch；同时仍保留可观的 FTV 外纵向信息。原始定义来自 site radiologist MRI report，参见同源 [npj Breast Cancer 方法](https://www.nature.com/articles/s41523-020-00203-7)。

## 3. Screening evidence

既有 [Radiomics Target Screening 报告](../../radiomics_target_screening/reports/final_report.md) 已冻结并直接复用：LD static FTV median |Spearman|=0.596，ΔLD–ΔFTV=0.369，Δ residual variance ratio=0.921，within/total variance=0.457。既有 strict-DCE7 G3 probe 的 static LD / ΔLD macro Spearman 为 0.324 / 0.138，R² 为 -0.069 / 0.001：弱但非零，不能替代 crop observability 证明。

LD 字段 `LD_T0`–`LD_T3` 映射 T0–T3。精确 workbook 与来源文件没有明示单位，故为 `LD_UNIT_NOT_EXPLICIT`；0 值无法区分 complete response、non-measurable、below detection 或 encoding floor，故为 `AMBIGUOUS_ZERO_SEMANTICS`。没有把 LD 擅自换算为 mm。

## 4. Stage A containment protocol

严格 cohort 为 375 人 × 4 visit = 1,500 patient×visit。实际 cache contract 为 `[4,8,32,96,96]`：前 7 个 channel 是 DCE7，第 8 个是只供 preprocessing/audit 的 binary localization support，模型不得读取。固定 crop 为 `(Z,Y,X)=(32,96,96)` voxel；T0 released bbox center 按 normalized index coordinate 投影到后续 visit，crop 前不做 spacing harmonization，越界零填充。

full support 使用 full-resolution FTV inclusion region proxy；它不是 manual dense lesion segmentation。legacy origin 未保存，因此在 clean candidate 各轴 ±2 voxel 内，要求 full mask crop 与 actual cached support bitwise exact reconstruction。1,500/1,500 visit 的 DCE 与 mask 在实际 reader index order 下 shape、spacing、slice-first handling 一致；mm 量是 matched-spacing index-space geometry，不是 affine/world-space registration 声明。

主指标与 gate 在看结果前冻结：suspected=`boundary touch OR containment<0.99 OR unavailable`，severe=`containment<0.90`，sufficient=`containment≥0.99 AND no touch`。独立 review 在首轮 aggregate 尚未生成时增加、不改变 gate 的保守 sensitivity：exact full-support retention、full-bbox retention、cached/full approximate maximum-extent retention、origin ambiguity 和 unique-only LD–margin rho。

## 5. Crop containment 结果

| Visit | n | Boundary touch | Suspected | Severe | Sufficient | Exact full support | Median margin mm | Q05 margin mm | LD zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 375 | 70.7% | 70.7% | 37.1% | 29.3% | 32.0% | -6.00 | -30.00 | 0.00% |
| T1 | 375 | 82.7% | 84.3% | 53.6% | 15.7% | 17.1% | -15.12 | -42.51 | 1.33% |
| T2 | 375 | 78.1% | 84.3% | 61.6% | 15.7% | 16.5% | -15.00 | -42.90 | 16.53% |
| T3 | 375 | 79.2% | 86.1% | 66.1% | 13.9% | 16.3% | -15.35 | -45.76 | 32.53% |

合并结果：boundary touch 77.7%，suspected truncation 81.3%，severe truncation 54.6%，sufficient containment 18.7%。更保守的 exact full-support retention 仅 20.5%；whole-union approximate extent retention 中位数 0.850，第 5 百分位 0.542。

origin bitwise reconstruction 为 99.2%，其中 unique 99.2%、ambiguous 0.0%；12 个 cached support complete miss 保守计为 ratio=0/severe。T0/T1 large-LD top quartile suspected truncation 为 99.0%，exact full-support retention 仅 1.0%；top 10% suspected truncation 为 100.0%，exact retention 为 0.0%。这构成大病灶系统性 mismatch 的强证据。

T0/T1 `Spearman(reported LD, minimum margin)` 为 -0.356，仍高于冻结 safety threshold -0.40；由于所有 matched origin 均 unique，unique-only sensitivity 相同。full-support largest-component approximate extent 与 LD 的 rho=0.599，只作 rank sanity；FTV proxy 与 radiologist target lesion 不等价，且 LD 单位不明，因此不作数值校准。

## 6. Stage A gate

| Criterion | Observed | Frozen requirement | Result |
|---|---:|---:|---:|
| `all_visit_sufficient_containment` | 0.187 | >= 0.850 | FAIL |
| `exact_origin_recovery` | 0.992 | >= 0.990 | PASS |
| `t0_t1_ld_margin_systematic_association` | -0.356 | > -0.400 | PASS |
| `t0_t1_suspected_truncation` | 0.775 | <= 0.100 | FAIL |
| `t0_t1_top_quartile_suspected_truncation` | 0.990 | <= 0.200 | FAIL |

三项关键 gate 失败，决策为 **NO-GO**，stop code 为 `LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP`。exact origin recovery 与 rho safety 通过，不足以抵消 containment failure。阈值没有因结果而放宽。

## 7. Conditional Stage B 方法

未执行。预注册方案本来只比较 B0（DCE7 JEPA）、B1（FTV-only）与 B2（FTV+LD），优先复用 matched G1/G3 controls，并以 2 seeds × 5 folds 运行 B2。Stage A NO-GO 后，该授权失效；没有 dual-grounding smoke、checkpoint、feature 或 prediction。

## 8. `lambda_LD` 选择

不适用。候选 [0.05, 0.1, 0.25] 没有启动，也没有读取 validation/test/pCR；`lambda_LD` 未选择。

## 9. Static FTV

不适用。没有 B2，不能计算 B2−B1 static FTV effect；既有 B1 结果只构成背景，不能被冒充为本实验 paired control 结果。

## 10. Static LD

不适用。没有 B2，无法回答 dual grounding 是否提高 LD decodability。Stage A 的 geometry/LD rank audit 不是 representation probe。

## 11. ΔFTV

不适用。训练未启动，未生成 frozen B2 state，也未运行统一 Ridge `Δr→ΔFTV`。

## 12. ΔLD

不适用。不能从既有弱 ΔLD probe 推断 static LD grounding 会自然改善 observed ΔLD；这仍是未检验假设。

## 13. FTV–LD trade-off

不适用。没有 B2−B1 的 static FTV、static LD、ΔFTV、ΔLD 四轴 paired effect，因此既不能宣称互补，也不能宣称 shared-state competition。

## 14. Optimization safety

不适用。没有 dual-target loss、gradient norm、representation std 或 base-loss trajectory；本轮既没有观察到新增 instability，也没有证据声称安全。禁止的 PCGrad、warm-up、two-stage 与 gradient normalization 均未加入。

## 15. pCR secondary evaluation

未执行且 Stage A 没有读取 pCR。由于没有 B2 frozen image-only representation，不存在可比较的 AUROC/AUPRC；pCR 方向变化为 `NOT_EVALUATED`，不能用缺失结果作正面或负面证据。

## 16. Limitations

1. FTV inclusion support 是 observability proxy，不是 radiologist LD target lesion 的 manual dense segmentation；multifocal whole-union 可能大于单一 target lesion。
2. legacy builder source/origin 缺失；12/1,500 visit 无法从 empty cached support 恢复 origin，但 origin-independent containment ratio 仍可精确计算。
3. 物理量基于 DCE-mask matched spacing 的 index-space geometry；没有做 affine/world-space registration inference。
4. approximate maximum extent 使用固定方向 extrema，而非 exhaustive all-pairs exact Feret；retention ratio 显式限制到 [0,1]，仅作 sensitivity。
5. LD 单位和 zero semantics 未确认；T2/T3 floor 会削弱后期 target，但没有用于放宽 gate。
6. 本报告不能回答任何 Stage B performance、optimization 或 clinical utility 问题。

## 17. Final decision 与 A–J

- **A. 当前 crop 是否真正能够观察 LD？** 不能可靠保证。T0/T1 suspected truncation 77.5%，exact full-support retention 仅 24.5%。
- **B. 大病灶是否存在系统 truncation？** 是。T0/T1 top-quartile 为 99.0%，top 10% 为 100.0%；后二者 exact retention 分别仅 1.0% 与 0.0%。
- **C. LD=0 floor 对 T2/T3 有多大影响？** T0 0/375（0%）、T1 5/375（1.33%）、T2 62/375（16.53%）、T3 122/375（32.53%）；T2/T3 明显，但语义仍 ambiguous。
- **D. FTV+LD 是否增强 LD decodability？** 未评估；Stage B 被 gate 阻止。
- **E. Static LD grounding 是否改善 observed ΔLD？** 未评估，不能外推。
- **F. FTV / ΔFTV 是否保持？** 未评估。
- **G. 双 grounding 是否加剧 JEPA/base instability？** 未评估。
- **H. Image-only pCR 是否有一致方向变化？** 未评估；pCR 未读取。
- **I. LD 是否提供第二 response axis？** 在当前 input 下未建立。table screening 的互补性不能越过 target observability failure。
- **J. 下一步做什么？** 先修改 input architecture，而不是扩 target 或优化 dual loss。

最终科学结论：**LD 在 table-level screening 上仍合理，但 current lesion-centered fixed crop 无法保证 target observability；因此不得训练 FTV+LD dual grounding，也不得声称第二 response axis。**

## 18. 下一阶段建议

优先级如下：

1. 采用覆盖 full lesion bbox 且保留明确 context margin 的 adaptive crop；
2. 或改为 spacing-aware larger physical FOV，避免同一 voxel crop 对应 30–128 mm 级别不等的实际视野；
3. 或构建 lesion-detail + breast/context multi-scale representation；
4. 修改 input 后原样重跑 outcome-free Stage A gate，仍不得因希望进入训练而降低阈值；
5. 只有新 input 通过后，才按冻结 B0/B1/B2、fold-train-only transforms、2 seeds × 5 folds 方案测试 LD。不要自动转向 SPH/BPE：SPH 需要独立 full-surface containment，BPE 需要对侧乳腺输入。

机器可读结论见 `metrics/final_decision.json`；阶段执行状态见 `metrics/stage_execution_status.csv`；公开隐私与哈希验收见 `metrics/public_artifact_verification.json` 与 `metrics/artifact_manifest.csv`。
