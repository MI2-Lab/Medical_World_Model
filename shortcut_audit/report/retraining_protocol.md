# CoRe-WM 五折审计重训练协议

## 1. 协议身份

项目方已明确允许在缺少原 checkpoint 的情况下自行重训练。因此后续实验定义为：

> 基于 `feature/ispy-clean-corejepa` 当前模型设计和 preprocessing 的五折
> **audit retraining**。

它不是原论文 checkpoint 的数值复现。所有表格、图和最终结论都会使用“审计重训练”
措辞，不再把结果与无法取得的原 checkpoint 作逐值一致性声明。

## 2. 五折与数据隔离

使用已验证的 seed-2026 long-format manifest：

```text
/data/data/Preprocessed/I-SPY2/
  _matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/
  matched_patient_cv_splits_seed2026.csv
```

SHA256：

```text
143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38
```

808 名 I-SPY2 患者每人恰好一次进入 held-out test。每折执行：

1. CoRe-JEPA representation：只用该折 primary train 加 156 名 I-SPY1 做
   pCR-free 训练；该折 validation 只做无监督 checkpoint selection；test 不参与；
2. response-target transform：只在 `pretrain_train` 上拟合；
3. frozen readout：只在 primary train 拟合；超参数和 threshold 只用 validation；
4. held-out test：只生成最终 prediction 和 audit perturbation，不用于任何选择；
5. 五折 test prediction 合并为唯一 OOF 结果。

每折模型 seed 为 `2026 + fold`，输出目录独立，checkpoint payload 保存 manifest
哈希、fold、patient order 和 split indices。

## 3. 影像 preprocessing 版本

本次影像输入明确采用现有完整 legacy cache：

```text
/data/data/Preprocessed/I-SPY2/
  _mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_
  autoroi_t0fallback_minfrac05_z32_y96_x96
```

它覆盖 808 名 I-SPY2 与 156 名 I-SPY1。audit adapter 只读取 `x [4,8,32,96,96]`，
并用 clean `mask_geometry(x[:,7])` 现场重算 q，不读取旧 label 或旧 geometry。

该 cache **不是**目标分支 clean `build_patient_tensor` 的逐例等价产物。全量 ROI crop
核对发现 3,232 个 I-SPY2 visit 中 154 个不同，涉及 77 名患者，主要来自半体素投影
后的临界舍入和一体素 crop 位移；其余一致病例仍可能有约 1 ULP normalization 差异。
详细证据见 `report/cache_compatibility.md`。因此最终报告将此版本称为
`legacy_adaptive_axiscanon_v1_autoroi_t0fallback_minfrac0.5`，不会称作 clean cache。

用户已授权只参考 repo 模型设计并自行重训练，故选择该版本作为本次实验的 canonical
preprocessing，以避免复制/重建约 36 GB 图像。106-D response features 没有兼容旧
缓存，使用 clean `response_targets.py` 重新提取。正式 response cache 已构建并通过
严格校验：`x_visit [964,4,106]`、clean raw target `[964,3,18]`，患者与特征顺序均
逐项一致，SHA256 为
`87698b7cd4f7d0130c30a6dac58958948dc094e29f3659f646ee2dd7ea120ac0`；提取过程不读取
pCR。完整机器可读记录见 `metrics/response_cache_validation.json`。

## 4. 模型和训练配置

模型、loss、optimizer、epoch、patience、EMA 和主要超参数原样取自
`ispy_jepa_tmi_clean/configs/paper_v1.yaml`。审计配置位于：

- `shortcut_audit/configs/retrain_paper_v1.yaml`
- `shortcut_audit/configs/audit_protocol_v1.yaml`

不会修改 `ispy_jepa_tmi_clean/` 核心代码；显式五折、缓存适配、readout threshold
和审计入口均放在 `shortcut_audit/`。

五折正式训练已完成。五折均运行 7 个 epoch，并由无监督 validation prediction 在
epoch 3 选择最佳 checkpoint；每折 validation `visit_state_std` 全程高于 0.05 的
防塌缩门槛。五个 held-out test 集合合并后为 808/808 个唯一 I-SPY2 患者。每折
seed、split size、最佳 loss、latent 标准差与 checkpoint SHA256 见
`metrics/fivefold_training_validation.json`。

## 5. Readout 与 threshold

每折使用 class-balanced logistic regression。penalty/C 只根据 validation 的三个
decision point 加权 AUROC选择。随后每个 decision point 仅用该折 validation prediction
选择最大 Youden J（等价于最大 balanced accuracy）的 threshold；并列时选择最接近
0.5、再选择较小者。test label 不进入拟合、超参数或 threshold 选择。

这与 repo 当前固定 0.5 threshold 有差异，原因是原审计规格明确要求
validation-selected threshold；两者都会在 provenance 中记录，不能混称。

## 6. 关键解释边界

当前 primary FLR 只读取由 geometry 与 clinical/treatment/time condition 生成的
`FutureResponseState`，不直接读取 MRI latent。因此：

- primary pCR 的 C1 MRI-only replacement 在架构上不会改变概率；
- temporal/donor primary pCR 变化只可能来自同步交换的 geometry；
- MRI trajectory 是否学习演化，应结合 JEPA latent 的 copy-current、native-target
  固定的 temporal/donor latent error 和另行定义的 static-T0 imaging readout判断；
- 不能把 geometry-only FLR 的零 MRI 敏感性表述成 MRI encoder 没有信息。

最终结论必须联合 B–F，不根据单一结果下判断。
