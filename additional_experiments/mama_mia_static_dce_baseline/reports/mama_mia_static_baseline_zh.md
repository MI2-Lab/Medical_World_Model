# MAMA-MIA-Inspired T0 静态 DCE-MRI pCR Baseline 实验报告

## 1. 一句话结论

在锁定的 808 人、seed-2026 五折上，T0 subtraction lesion-centered R3D-18 得到 AUROC 0.523、AUPRC 0.366、balanced accuracy 0.517；T0 pre/post1/post2 得到 0.549、0.396、0.528。三相输入相对 subtraction 有小幅数值提升但配对 bootstrap 置信区间跨 0；在严格对齐的 375 人 radiomics complete-case cohort 上，三相影像与已有 radiomics 基本相当。静态信号弱但非完全缺失，下一步值得在不改变 split 和 T0 表示的条件下检验 T1 的增量价值。

## 2. 研究问题

本实验只回答治疗前单次检查能提供多少 pCR 信号。同一次 T0 扫描内的增强过程（pre→post1→post2）是 **within-scan DCE dynamics**，不等于治疗过程中 T0→T1 的 **longitudinal treatment dynamics**。本轮任何模型均不使用 T1/T2/T3。

## 3. 数据与 cohort

正式 population 为现有 `matched_patient_cv_splits_seed2026.csv` 的 808 名 I-SPY2 患者，pCR 275/808（34.03%）。manifest SHA-256 为 `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。

Phase audit 结果：

| 项目 | 数量 |
|---|---:|
| 总患者 | 808 |
| 完整 pre/post1/post2 | 808 |
| 缺 phase | 0 |
| 缺 pCR label | 0 |
| 缺 T0 ROI | 0 |
| 最终纳入 | 808 |

BreastDCEDL metadata 中所有纳入患者 `pre=0`；本实验据此明确定义 `post1=pre+1=1`、`post2=pre+2=2`，而不是按文件名排序推断。三帧来自 manifest 指定的同一个 4-D T0 NIfTI，天然共享 affine、orientation、voxel grid 与 shape，无需重复 registration。15 例 dcm2niix partial-z acquisition 使用 manifest 已指定的 `original_DCE_aligned.nii`，而不是未修复的固定文件名。

| Fold | Train N | Val N | Test N | Train pCR % | Val pCR % | Test pCR % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 525 | 121 | 162 | 33.90 | 34.71 | 33.95 |
| 1 | 525 | 121 | 162 | 34.10 | 33.88 | 33.95 |
| 2 | 525 | 121 | 162 | 33.90 | 34.71 | 33.95 |
| 3 | 526 | 121 | 161 | 34.03 | 33.88 | 34.16 |
| 4 | 526 | 121 | 161 | 34.03 | 33.88 | 34.16 |

每折 train/validation/test 患者集合两两交集均为 0；每名患者恰好一次进入 outer test。

## 4. 方法

### 4.1 输入与 crop

- Baseline A：`post1 - pre`，复制到 3 channel，以完整复用预训练 backbone。
- Baseline B：`[pre, post1, post2]` 作为 3 channel，z/slice 作为 video temporal dimension，输入为 `[B,3,Z,H,W]`。
- 定位仅使用人工 T0 FTV lesion ROI。本 baseline 因此测量的是 **lesion-localized image representation** 的 pCR prediction capability。
- 两个 baseline 使用完全相同的 T0 ROI bounding box；各轴扩张 25%，再线性重采样为 `32×96×96`。mask 只用于 crop/QC，不作为模型 channel。

### 4.2 标准化、模型与训练

对每名患者，仅用其 T0 pre crop 的正值 voxel 计算 1%/99% 截断、中位数与 robust IQR scale；同一个 reference 应用于 pre/post1/post2，最后截断到 `[-5,5]`。没有使用 validation/test cohort statistics。

模型为 TorchVision Kinetics-400 V1 预训练 `r3d_18`，仅替换末端为单 logit。两模型统一使用 AdamW（lr `1e-4`，weight decay `1e-4`）、train-fold-only `pos_weight` BCE、batch size 16、最多 20 epoch、validation AUROC→AUPRC 选择 checkpoint、patience 5、validation-only Youden J threshold。增强仅为训练期 x/y 随机翻转。每折 seed 为 `2026+fold`。

### 4.3 评价与不确定性

主结果为五折 pooled OOF。AUROC、AUPRC 与 balanced accuracy 使用固定 seed 2026、5,000 次 patient-level bootstrap；A/B 共用完全相同的 patient bootstrap indices，ΔDCE 因而为 paired bootstrap。分类阈值只来自对应 fold validation。

## 5. 实验结果

| Model | Population | AUROC (95% CI) | AUPRC (95% CI) | Balanced Acc. (95% CI) |
|---|---:|---:|---:|---:|
| Existing Clinical-only | 808 | 0.709 | 0.558 | N/A |
| Existing Radiomics-only | 375 | 0.564 | 0.350 | N/A |
| T0 subtraction lesion-centered | 808 | 0.523 (0.481–0.565) | 0.366 (0.321–0.421) | 0.517 (0.481–0.553) |
| T0 3-phase DCE lesion-centered | 808 | 0.549 (0.506–0.592) | 0.396 (0.344–0.452) | 0.528 (0.491–0.565) |

Clinical-only 与两种影像模型共用 808 人和相同 outer folds，但旧表未保存本报告需要的 balanced-accuracy threshold 结果，故标记 N/A。Radiomics-only 只适用于预先锁定的 375 人 complete-case population，不能把其绝对值与上表 808 人影像绝对值当作严格比较。

Baseline A 的 AUROC 与 balanced-accuracy CI 均跨 0.5，AUPRC CI 也覆盖 prevalence 0.340 附近；因此 Q1 的结论是：subtraction 没有稳定优于 chance 的证据，不能描述为强 predictor。

## 6. DCE incremental value

定义 `ΔDCE = Perf(T0 pre/post1/post2) − Perf(T0 subtraction)`：

| Metric | ΔDCE | Paired bootstrap 95% CI |
|---|---:|---:|
| AUROC | +0.0265 | −0.0194–+0.0700 |
| AUPRC | +0.0306 | −0.0189–+0.0756 |
| Balanced Accuracy | +0.0110 | −0.0288–+0.0504 |

Q2 的答案是否定的：当前数据和协议下，保留 pre/post1/post2 相对 subtraction 没有统计上清晰的增量。

### 与 radiomics 的同 population 比较

| Model（同一 375 人） | AUROC | AUPRC | Balanced Acc. |
|---|---:|---:|---:|
| Existing Radiomics-only | 0.564 | 0.350 | N/A |
| T0 subtraction | 0.511 | 0.318 | 0.506 |
| T0 3-phase DCE | 0.568 | 0.359 | 0.542 |

Q3 的答案是：三相影像与 radiomics 基本相当，数值上高 0.0038 AUROC、0.0092 AUPRC，差异小到不足以声称超过。旧 radiomics 结果与本实验使用同一锁定患者集合和 outer folds，但其患者级预测未纳入公共仓库，因此本轮只能作同 population 的绝对 OOF 比较，不能声称有 paired CI 或显著性。

## 7. Sanity / leakage audit

| 检查 | 结果 |
|---|---|
| Random-label | Three-phase fold 0 单次 run：test AUROC 0.373，AUPRC 0.269，balanced accuracy 0.392；未出现伪高性能 |
| Zero-image | pooled AUROC：subtraction 0.501，three-phase 0.500；相对正式模型退化 |
| ROI visual QC | 固定 seed 随机 6 例均显示病灶位于 crop，subtraction 突出增强区；见 `figures/t0_crop_qc.png` |
| Split audit | 5/5 folds 的 train/val/test 交集均为空；808 人各一次 outer test |
| Phase/grid audit | 808/808 同一 T0 4-D acquisition 内三个 frame，完整且同 grid；15 例按 manifest 使用 partial-z 修复文件 |
| Longitudinal leakage | loader 只构造 `/<patient>/T0/` 路径；cache 文件仅含 T0 三帧、T0 mask、label 与 normalization metadata；未读取 T1/T2/T3、future measurement、clinical、treatment 或 radiomics |

另发现源数据 header 问题：54/808 的 DCE affine、36/808 的 orientation code 与 ROI header 不一致。经核验，manifest-selected DCE 与 ROI 的 array shape 及 manifest bbox/实际 mask bbox 在 808/808 完全一致；本实验因此明确使用 preprocessing manifest 已映射到 DCE voxel-index grid 的 bbox，不执行不可靠的 affine 投影。这是已披露的 preprocessing header limitation，不影响同一 4-D DCE 内三个 phase 的互相对齐，但限制 physical-space 解释。Zero-image 在不同 folds 的常数概率可因 checkpoint bias 不同而变化，但每个 fold 内所有患者输入完全相同；pooled AUROC 约 0.5，符合预期。

## 8. 与 MAMA-MIA 的关系

Geissler & Schäfer（2026，arXiv:2608.29162）的任务同样是 pretreatment DCE-only pCR prediction，并采用 lesion-centered pre + first two post volumes 和预训练 video classifier；其 25-model ensemble 报告 balanced accuracy 约 0.541、AUC 约 0.572。我们的单模型结果与其量级接近，但 dataset、split、domain 与 evaluation protocol 均不同，不能视为公平 benchmark。

该工作的启示不是追求大型 ensemble，而是：baseline morphology/early enhancement 含有限信号，同时留下了检验治疗期 longitudinal response 的空间。

## 9. 对 Cancer World Model 的意义

本结果属于“静态 T0 DCE 较弱到中等”的情形。三相 DCE 没有显著超过 subtraction，且在 375 人配对 population 仅与 radiomics 基本相当。可能原因包括有限样本、跨站点 acquisition timing/强度异质性、ROI 定位与 25% bbox crop、源 header 异质性，以及 pCR 生物学不能由基线形态充分决定。按照预注册原则，不因结果偏低继续 architecture/crop search。

更重要的是，未来 World Model 不能只与弱的单相或全乳背景模型比较，而应把这里的三相 lesion-centered T0 作为静态 comparator，直接估计：

`Δlongitudinal = Perf(T0+T1) − Perf(T0 DCE phases)`。

## 10. 下一步实验建议

值得进入 T0+T1 实验，但应保持本轮 808 人 manifest、T0 pre/post1/post2 表示、T0 ROI crop、normalization、readout/evaluation 和 bootstrap indices不变，只加入 early-treatment T1。先检验 paired ΔAUROC/ΔAUPRC；只有 T1 显示稳定增量后，再考虑 T2/full World Model。

## 11. 可复现资产

- 冻结配置：`configs/formal.yaml`
- Phase/split 审计：`metrics/phase_audit_summary.json`、`metrics/split_summary.csv`
- Grid/header 审计：`metrics/geometry_alignment_audit.json`
- 完整交付验证：`metrics/formal_validation.json`
- 每折指标与 checkpoint metadata：`metrics/per_fold_metrics.csv`、`metrics/best_checkpoint_metadata.json`
- Bootstrap 汇总：`metrics/summary.json`
- Sanity：`metrics/random_label_test.json`、`metrics/zero_image_test.json`
- 运行环境与输入哈希：`metrics/run_provenance.json`
- QC：`figures/t0_crop_qc.png`
- 患者级 val/test prediction、bootstrap indices、phase manifest、训练日志与 checkpoints 已保存在实验目录对应子目录，并按 repository privacy policy 通过 `.gitignore` 排除，不提交患者标识或大型 binary。
