# DINOv3 与 Radiomics Grounding：Medical World Model 下一步建议

日期：2026-08-31
范围：`Medical_World_Model*/additional_experiments` 相关实验
目的：评估 DINOv3 是否能够改善 image representation，并判断 radiomics grounding 是否有希望解决 Core-WM 中 clinical data 主导、image feature 缺乏 pCR 增量的问题。

## 1. 执行摘要

建议继续探索 DINOv3，但不建议重复当前的“冻结 DINOv3 特征后直接与 clinical 拼接”的方案。

现有结果支持以下判断：

1. DINOv3-LOCAL 能够编码 FTV 及部分纵向 FTV change，说明它确实提取到了有意义的影像表型。
2. DINOv3 的 MRI-only pCR 表现比原来的 LOCAL0 有一定改善，尤其是 T0–T1；但结果仍不稳定。
3. 直接把 1536 维 DINOv3 feature 与 clinical 拼接后，pCR 表现反而下降，主要问题很可能是高维 fusion 的有限样本过拟合和缺乏条件式融合，而不是 DINOv3 完全没有影像信息。
4. 当前 DINOv3 实验实际上是冻结的 2D slice feature baseline：只使用 7 个 DCE 通道中的 3 个，并对 32 张 slice 做平均。因此它不能否定经过 MRI-domain adaptation 的 DINOv3。
5. 最合理的下一步是：以 confirmed LOCAL3 作为 Core-WM 对照，训练一个低容量 MRI adapter，将 DINOv3 适配到 DCE7/3D longitudinal input，再把 radiomics supervision 直接加在 observed image state 上，最后用 clinical-offset/低维 late fusion 检验真正的 conditional image increment。

这里的目标不是让 image feature 在模型系数中看起来比 clinical 更重要，而是证明在已知 clinical、treatment 和 FTV 后，image 是否仍提供独立信息。

## 2. 当前 DINOv3 实验做了什么

现有 posthoc DINOv3 使用 `facebook/dinov3-vitb16-pretrain-lvd1689m` 的冻结 ViT-B/16 backbone：

- 约 85.7M 参数，representation dimension 为 1536；
- 每个 visit 使用 `early`、`late`、`late_minus_pre` 三个通道；
- 32 张 axial slice 分别输入 224×224 2D ViT；
- 每张 slice 使用 CLS 与 patch mean 的拼接；
- 最后对 32 张 slice 的 feature 做 arithmetic mean；
- 同时测试 GLOBAL 和固定 64mm LOCAL；
- extraction 本身不读取 outcome，pCR 只用于后续 frozen probe。

因此，这个实验的主要限制是：

- 没有使用完整 DCE7 动力学信息；
- 没有真正的 3D encoder 或跨 slice temporal aggregator；
- dense patch tokens 在 per-visit aggregation 前已被强烈平均；
- backbone 没有针对 MRI 或 breast cancer 做 adaptation；
- formal aggregate report 目前还没有提交到仓库，结果来自本地 private features/predictions。

## 3. 现有结果

以下数字是从保存的 OOF predictions 重新聚合的 exploratory 结果。它们不是新的训练结果，也不应替代正式的 5-seed confirmation。

### 3.1 DINOv3 的 image-only pCR 信号

full 808 cohort 上，DINOv3 LOCAL 的 MRI-only OOF AUROC 为：

| Decision point | DINOv3 LOCAL MRI-only AUROC |
|---|---:|
| T0 | 0.531 |
| T0–T1 | 0.605 |
| T0–T2 | 0.590 |

相比当前旧的 LOCAL0，DINOv3 LOCAL 在 T0–T1 的点估计约高 0.045，但 patient-level paired bootstrap CI 仍接近跨零。因此可以说有 promising signal，但还不能说 DINOv3 已经稳定改善 pCR。

### 3.2 DINOv3 对 FTV 的表征能力

在 375 名 radiomics-complete patients 上：

- T0 FTV：Spearman ρ = 0.813，R² = 0.522；
- T0→T1 ΔFTV：Spearman ρ = 0.439，R² = 0.110。

这说明 DINOv3 的影像表示确实能捕获肿瘤大小及其变化，且 LOCAL 明显优于 GLOBAL。例如 T0→T1 的 pCR AUROC 中，DINOv3 LOCAL 相对 GLOBAL 的差异约为 +0.104，paired bootstrap CI 为 [+0.054, +0.157]。

但这也带来一个重要问题：DINOv3 可能主要捕获了 FTV-related signal，而不是 FTV 之外的独立治疗反应信号。

### 3.3 与 clinical 的融合

clinical-only AUROC 为约 0.709。DINOv3 LOCAL 与 clinical 直接融合后：

| Decision point | Clinical-only | Clinical + DINOv3 LOCAL |
|---|---:|---:|
| T0 | 0.709 | 0.693 |
| T0–T1 | 0.709 | 0.657 |
| T0–T2 | 0.709 | 0.637 |

T0–T1 和 T0–T2 相比 clinical-only 的 paired AUROC 差异约为 -0.053 和 -0.072，说明当前直接 fusion 不能证明影像带来增量，反而很可能放大了高维过拟合。

在 375 名 patients 上，加入 FTV 后的 clinical + FTV + DINOv3 也没有改善 clinical + FTV，T0/T0–T1/T0–T2 的差异约为 -0.069/-0.103/-0.133。

这与已有 compact fusion audit 一致：原始高维 fusion 的 train–test gap 很大；降维可以缓解过拟合，但没有稳定恢复 clinical- 或 FTV-complementary signal。

## 4. 对 PI 建议的具体判断

PI 建议试 DINOv3 是合理的，但应该把它定位为“更强 image representation 的候选”，而不是直接替代 Core-WM 或直接提升 pCR 的办法。

目前结果表明：

- DINOv3 方向没有被否定，因为它在 FTV 和部分 MRI-only pCR probe 上有信号；
- 但 off-the-shelf frozen DINOv3 并没有解决 clinical dominance；
- 旧版 DINO ImageNet feature 在部分 FTV probe 上甚至强于 DINOv3，说明模型版本本身可能不是主要瓶颈；
- 更关键的是输入适配、3D/longitudinal aggregation、observed-state grounding 和低维 conditional fusion。

因此，不建议继续做更大的 DINO 模型加 raw concatenation。应改成 MRI-adapted, radiomics-grounded, conditional evaluation。

## 5. 建议的下一轮实验

### 5.1 固定实验臂

建议至少保留以下四个 arm：

| Arm | 方案 | 用途 |
|---|---|---|
| A | Core-WM confirmed LOCAL3 | 固定现有最佳 baseline |
| B | 当前 frozen DINOv3 LOCAL | 复现现有 posthoc 结果 |
| C | DINOv3 + MRI adapter，无 radiomics grounding | 分离 domain adaptation 的作用 |
| D | DINOv3 + MRI adapter + radiomics grounding | 检验 PI 提出的核心假设 |

可以额外加入 `C + FTV-only grounding`，用来区分“只学到 FTV”与“学到 FTV 之外的 phenotype/appearance”。

第一阶段可以使用 2 seeds × 5 outer folds 做筛选；只有通过 gate 的方案再进行 5-seed confirmation。

### 5.2 DINOv3 的输入和 adapter

第一版应控制复杂度：

- 保留 64mm LOCAL 作为主输入，不重新引入表现不稳定的 full-context branch；
- 增加小型 learnable `7→3` channel adapter，使完整 DCE7 能进入 DINOv3；
- backbone 初始保持 frozen；
- 训练一个低容量 slice aggregator，保留 axial position，而不是简单平均 32 张 slice；
- 如果 adapter 版本通过 gate，再测试 LoRA 或只解冻最后 1–2 个 blocks；
- 不建议一开始 full fine-tune。

之前 patch-token WM 直接预测 treatment-conditioned token change 已经失败。因此新的 token 使用方式应限制为：利用 dense/local tokens 构建 observed image state，再做 radiomics grounding；不要直接训练一个高容量 token-level pCR/world-model predictor。

### 5.3 Radiomics grounding 的目标

现有 M2 的主要问题是：radiomics head 接在 predicted image delta 上，导致绝大多数 target prediction near-constant。新方案应改为：

- 在真实 observed state `z_t` 上预测静态 radiomics；
- 在 `z_{t+1} - z_t` 上预测 observed radiomics change；
- 不要只在 predicted future state 上监督；
- mask 只用于生成 target，不作为 image encoder 输入；
- 增加 mask-only/geometry-only control，排除 ROI 几何 shortcut。

现有四个低维 measurement 是 FTV、SPH、LD、BPE，并不是完整的 high-dimensional PyRadiomics texture set。建议第一阶段使用少量、可观测、稳定的 appearance/kinetic features，而不是直接加入大量纹理特征。

target 选择应优先依据：

1. mask perturbation 或技术重复下的稳定性；
2. 与 FTV 的非冗余性；
3. 在当前 crop 中的物理可观测性；
4. 预测后是否出现 non-collapse。

不应使用全体 pCR 结果事后挑选 radiomics target。若要做 residual target，可在每个 outer-train 内拟合：

```text
R_residual = R - E[R | log(FTV)]
```

这样更直接地测试 image 是否学习 FTV 之外的 phenotype information。

LD 和 BPE 暂不应直接训练：LD 在当前 crop 下不可观测，BPE 又缺少可靠的 contralateral source ROI。它们需要先通过新的 context/observability audit。

### 5.4 Clinical 融合方式

不要将 1536 维 DINO feature 和 clinical raw variables 直接 concatenate。建议使用 cross-fitted low-dimensional score 或 clinical offset：

```text
clinical_score = f_clinical(C)
image_score    = f_image(z_DINO)

logit P(pCR) = clinical_score + beta * image_score
```

必须报告两个 estimand：

- `clinical + image` 相比 `clinical`；
- `clinical + FTV + image` 相比 `clinical + FTV`。

第二个比较尤其重要，因为第一个可能只是 image 复现了 FTV 或其他临床相关因素。

## 6. 建议的 GO/NO-GO 标准

Radiomics-grounded DINO 只有在以下条件同时得到支持时，才值得进入完整 WM：

- 相比无 grounding 的 DINO adapter，held-out FTV/ΔFTV 和 residual radiomics decodability 提升；
- MRI-only pCR 在 T0–T1 或 T0–T2 有一致正向趋势；
- `clinical + image` 不再系统性低于 `clinical`；
- `clinical + FTV + image` 至少不系统性低于 `clinical + FTV`；
- Brier/log-loss 不恶化；
- train–OOF gap 不增加；
- 5 seeds 中至少 4 个方向一致。

如果 DINOv3 只改善 radiomics/FTV 解码，却没有 clinical-conditional pCR 增量，应将其结论限定为“更好的 image state representation”，不要继续为 pCR 增加模型复杂度。

## 7. 实验顺序

推荐执行顺序：

1. 完成现有 DINOv3 private results 的正式 aggregate report 和 reproducibility handoff；
2. 用相同 fold、population、timing 和 readout 复现 A/B；
3. 做 frozen DINOv3 + 7→3 adapter + 低容量 slice aggregation；
4. 加入 FTV-only grounding 作为 control；
5. 再加入少量 FTV-residual appearance/kinetic radiomics；
6. 通过 conditional clinical-offset 评估 pCR incremental value；
7. 只有 image representation 和 conditional pCR 都出现稳定改善后，才把它接回完整 temporal WM。

不建议优先做：更多 patch-token WM、单纯增大 DINO、raw clinical concatenation、在当前 crop 下直接加入 LD、或继续大范围扫描 grounding loss weight。

## 8. 主要限制

- 当前结论主要基于内部 OOF，尚无 external validation；
- seed 不是独立患者样本，不能替代外部队列；
- 375 名 radiomics-complete patients 与 808 名 pCR cohort 并非完全可交换；
- T3 属于 late/pre-surgery timing，不能与 early treatment response 混为一谈；
- DINOv3 foundation artifacts 当前仍有大量 untracked/private 文件，正式报告和版本化交付尚未完成；
- pCR 是 treatment、subtype、clinical factors 共同决定的终点，clinical dominance 本身并不证明 image encoder 失败。

## 9. 参考文件

- [Core-WM radiomics/Next-Change report](additional_experiments/radiomics_next_change/reports/final_report.md)
- [Compact MRI–clinical fusion audit](additional_experiments/compact_mri_clinical_fusion_audit/reports/final_report.md)
- [Latest raw/spatial pCR ceiling](additional_experiments/raw_spatial_pcr_ceiling/reports/final_report.md)
- [C1B eligibility and FTV Stage B](additional_experiments/c1b_overlap_eligibility_ftv_stageb/reports/final_report.md)
- Local DINOv3 extraction contract：`additional_experiments/foundation_mri_dinov3_posthoc/features/formal/dinov3_vitb16_lvd1689m_posthoc/contract.private.json`
- Local DINOv3 OOF predictions：`additional_experiments/foundation_mri_dinov3_posthoc/predictions/dinov3_baseline_predictions.private.csv`
