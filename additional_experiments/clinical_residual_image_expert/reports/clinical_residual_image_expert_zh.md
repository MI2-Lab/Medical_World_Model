# Clinical Residual Image Expert（T0 DCE）

## 1. 一句话结论

本次严格外层 OOF 实验的判定为 **NO COMPLEMENTARY T0 SIGNAL**：T0 MRI 在已使用 clinical 信息后没有显示出可信的患者特异性 pCR 增量；解释必须同时参考重校准和影像依赖审计，不能只看 AUROC 的单点差异。

## 2. 实验动机

Clinical-only 已是强预测器，普通拼接融合可能被 clinical dominance 主导。主模型固定 clinical 预 sigmoid logit，仅令 Kinetics 预训练 R3D-18 从冻结的 T0 三期 DCE 中学习校正项：`l = l_c + Δl_I`。残差头权重与 bias 均以零初始化，故第 0 步严格等于 clinical-only。

## 3. 数据与 split

808 名患者、275 名 pCR，使用锁定的 seed-2026 五折 patient-level outer CV（manifest SHA-256: `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`）。T0 输入直接读取既有 MAMA-MIA 缓存：pre/post1/post2、32×96×96、FTV bbox+25% 和冻结的 per-patient T0-pre 归一化；没有读取 T1/T2/T3 或重新做影像预处理。

## 4. Clinical baseline

特征为 HR、HER2、MammaPrint、年龄和 exact treatment arm。年龄只用 outer-train 均值填补，arm 只以 outer-train support one-hot，随后 StandardScaler 与 balanced liblinear Logistic Regression；C/penalty 仅由 validation AUROC→AUPRC 选择。复现实验 AUROC=0.709156、AUPRC=0.557514，相对锁定值的 AUROC 差=+0.000000。

## 5. Image residual architecture

`l = l_c + Δl_I`，训练为 `BCEWithLogitsLoss(l, y) + 1e-3·mean(Δl_I²)`。clinical 分支不在影像训练中更新。每个 outer train 患者的 `l_c` 来自五折 inner cross-fitting，因此并非 clinical in-sample prediction；validation/test 只使用 outer-train 拟合的 clinical model。

## 6. 主要结果

| Model | AUROC | AUPRC | Balanced Acc. | ΔAUROC vs Clinical | ΔAUPRC vs Clinical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clinical-only | 0.7092 | 0.5575 | 0.6611 | – | – |
| T0 image-only（既有） | 0.5492 | 0.3964 | 0.5281 | -0.1599 | -0.1611 |
| Clinical recalibration | 0.7244 | 0.5842 | 0.6611 | +0.0153 | +0.0267 |
| Naive Clinical + Image | 0.5712 | 0.4018 | 0.5524 | -0.1380 | -0.1557 |
| Clinical + Image Residual | 0.6895 | 0.5447 | 0.6538 | -0.0196 | -0.0128 |

## 7. Paired bootstrap

所有比较共享 MAMA-MIA 的同一组 patient-level seed-2026、5,000 次 bootstrap indices。Residual vs Clinical：ΔAUROC=-0.019635，95% CI [-0.039268, +0.000666]；ΔAUPRC=-0.012843，95% CI [-0.039813, +0.015054]。Naive fusion 的 ΔAUROC=-0.137957。

## 8. Recalibration control

不读取 MRI 的 logistic recalibration（`l'=a·l_c+b`）ΔAUROC=+0.015269。Residual 的 ΔAUROC=-0.019635，未超过该 control；必须与上述 paired CI 一起解释。

## 9. Image reliance audit

Residual 模型 real-image AUROC=0.6895；zero-image=0.6574；固定 patient-shuffled image=0.6695。real-image 高于两项反事实。

## 10. Correction analysis

clinical confident-but-wrong 组 n=34，平均 |Δl_I|=0.3181，有益/恶化校正=17/17。该分析是 post-hoc，不用于调参。

## 11. 结论

该诊断实验只检验 `I(T0 MRI; pCR | Clinical)>0?`。结论：**NO COMPLEMENTARY T0 SIGNAL**。若未同时优于 clinical recalibration 且不能通过 zero/shuffled image 审计，则不应把单点性能变化称为 MRI 的补充信息。
