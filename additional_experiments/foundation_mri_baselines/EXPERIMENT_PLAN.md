# DCE-MRI Foundation Encoder Baselines：冻结实验计划

## 1. 目标与判断边界

本实验在同一 I-SPY2 808 人、同一锁定五折、同一 pCR 标签和同一信息时点下，比较强公开预训练 encoder、当前 CNN response state、clinical 与 radiomics/FTV。主问题是 frozen representation 中有多少 pCR 相关影像信息，而不是大型 fine-tuning 后的最优预测性能。

正式 test 评估前冻结两个候选：

1. `medicalnet_resnet50_3dseg8`：3-D 医学影像参照；
2. `dino_vitb16_imagenet1k`：固定语料的通用自监督参照。

BiomedCLIP、DINOv2、DINOv3 因无法排除试验级 web/PMC 污染或许可/匿名获取条件更差，在查看任何 pCR test 结果前排除。正式候选不会按 test AUROC 过滤，两个模型的全部正式结果均报告。

## 2. 数据与 estimand

- Primary：正式 808 人，275 pCR 阳性；每人恰好一次 outer-test，汇总 808 个 OOF prediction。
- Complete-case sensitivity：375 人、110 pCR 阳性，只用于需要 FTV/radiomics 的 paired 比较。
- FTV 缺失为明显非随机；绝不把 375 人绝对分数当作 808 人 primary 的替代。
- I-SPY1 的 139 个 train-only 患者不进入本实验 probe；所有模型用完全相同的 I-SPY2 五折。
- patient-level representation/prediction 均为 ignored `*.private.*`、mode 600；公开文件只含聚合量、计数和哈希。

正式 fold SHA-256：

`143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`

## 3. C1B-H 与 DCE7

输入固定为 C1B-H `float32 [4,7,112,176,160]`，四个 visit 共用 T0 anchor/grid，XYZ spacing 为 `(0.9,0.9,2.0)` mm。DCE7 顺序为：

`pre, early, late, early-pre, late-pre, peak-relative-enhancement, late-minus-peak-relative-enhancement`。

C1B 已按 patient/visit/channel 在 valid-source 区域作 robust normalization 并截断至 `[-5,5]`。新 adapter 不访问 clinical、FTV、pCR、support/mask 或原始 sidecar。LOCAL 是固定物理中心 64-mm cube，不使用 lesion mask；准确表述为 mask-free，但继承冻结的 T0 localisation prior。

## 4. 冻结 encoder 与输入适配

### MedicalNet

- 严格载入论文时代 3DSeg-8 `resnet_50.pth`，只移除一个 `module.` 前缀；318/318 state entries、46,155,072 parameters 必须 100% 覆盖。
- 单通道 encoder 不作 7-channel inflation。七个 DCE channel 作为共享 frozen encoder 的七个 batch items，按固定 DCE7 顺序拼接。
- 保留 native C1B grid。layer4 必须为 `[7,2048,14,22,20]`。
- GLOBAL 为 layer4 空间均值；LOCAL 复用 hash-locked sampling-cell overlap primitive，在同一 layer4 map 上作固定中心 64-mm 加权均值。
- 每 visit 表示维度 `7×2048=14,336`。

### DINO v1

- 使用 Meta 原生 ViT-B/16 checkpoint 与原生 LayerNorm `eps=1e-6`，150/150 tensors strict load。
- 三通道在 test 前由 DCE 物理语义固定为 `early, late, late-minus-pre`；不按 pCR 选择 channel。
- GLOBAL：完整 Z 轴均匀重采样 32 张 axial planes，176×160 对称 pad 为 square，再显式 bicubic resize 224。
- LOCAL：固定中心 64-mm cube 采样为 32×72×72，再显式 bicubic resize 224。
- 所有插值完成后统一 clip `[-5,5]`、固定 affine 到 `[0,1]`、ImageNet mean/std；禁止 slice-wise min-max。
- 每 slice 使用官方线性评估特征 `concat(final CLS, mean(final patch tokens))`，再对 32 slices 固定均值；每 visit 维度 1,536。

两模型全程 `eval()`、`requires_grad=False`，正式 GPU extraction 固定 bf16 autocast，输出存为 float32。所有 checkpoint 与 extraction-contract hash 写入私有 sidecar 和公开执行账本。

## 5. 信息时点与线性 readout

唯一 primary 时点与 feature schema：

- T0：`z0`
- T0-T1：`[z0,z1,z1-z0]`
- T0-T2：`[z0,z1,z2,z1-z0,z2-z1,z2-z0]`

最后一项沿用当前正式 CNN pCR readout，确保 foundation 与 current CNN 完全同协议。详细禁止未来信息规则见 `configs/information_timing_contract.csv`。T0-T3 不是早期预测 primary，本轮不运行。

每 fold 的 StandardScaler 只拟合 outer-train。class-balanced logistic regression 的 penalty/C 只按 validation 选择，Youden threshold 也只由 validation 冻结。选择锁定后才构造 outer-test matrix，并以 single-use guard 调用一次 `predict_proba`。Primary 汇总 808 个 OOF prediction，报告 AUROC、AUPRC、Brier、calibration intercept/slope 与固定 10-bin ECE。

## 6. Baseline matrix

### 808 人 primary

- Clinical-only；
- current CNN `GAP0` MRI-only / +clinical；
- current CNN `LOCAL0` MRI-only / +clinical；
- MedicalNet GLOBAL/LOCAL MRI-only / +clinical；
- DINO GLOBAL/LOCAL MRI-only / +clinical。

`GAP0/LOCAL0` 使用 seed 2026、无 FTV supervision 的现有逐 fold 资产；不重训旧 CNN，也不修改旧实验。

### 375 人 paired sensitivity

- Clinical-only、FTV-only、Radiomics-only；
- Clinical+FTV、Clinical+Radiomics；
- Foundation+FTV、Foundation+Clinical+FTV；
- 对同一人、同一 fold、同一 timing 报告 beyond-FTV 增量。

## 7. Phenotype 与 FTV decodability

- Frozen T0 representation 以相同 train/validation/test discipline probe HR、HER2；可靠完整的 HR/HER2 subtype 另作 multiclass macro OVR probe。
- 在 375 complete cases 上用 outer-train scaler + validation-selected Ridge 解码 FTV 与相邻 ΔFTV；报告 R²、MAE、Spearman。
- Foundation pCR 强而 FTV 解码也强，只能说明可能更好编码体积；`Clinical+FTV` 对 `Clinical+FTV+Foundation` 的 paired 增量才是 beyond-FTV 主要证据。

## 8. Scope 与解释

Primary 是 frozen encoder + linear probe。Light fine-tuning 与 temporal Transformer 不运行，以避免把 representation-quality diagnosis 与大模型调参混为一谈。若所有 MRI-only 均弱而 clinical 强，结论是本任务的 MRI 增量有限；若 foundation 明显优于 current LOCAL，才支持当前 image encoder underuse phenotype；若 foundation 在 clinical/FTV 之上仍提升，才支持 complementary phenotype。

所有运行失败均写入 `reports/model_execution_ledger.md`；禁止静默删患者、替换 checkpoint 或只报告最好候选。

