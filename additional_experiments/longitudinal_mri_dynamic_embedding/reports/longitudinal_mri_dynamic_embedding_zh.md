# Longitudinal MRI Dynamic Embedding（T0–T3）

## 一句话结论

**NO CONFIRMED EARLY DYNAMIC MRI VALUE**。本实验以严格因果 temporal embedding 检验治疗中 MRI 动态是否提供 clinical 外 pCR 信息；主要临床决策点为 T0→T1。

## 设计

808 名患者、275 名 pCR、固定 seed-2026 五折 outer CV。每访视的三相 DCE（pre/early/late）由共享 Kinetics R3D-18 编码，再由一层 causal transformer 汇聚；位置 t 只能 attend T0..t。ROI mask、FTV、radiomics、geometry scalar 都没有输入。Residual 输出为 `l=l_clinical+Δl_dynamic`，clinical logits 在 outer-train 内五折 cross-fitting 后冻结。Clinical-only 复现 AUROC=0.709156、AUPRC=0.557514。

## 主要结果：T0→T1

| Model | AUROC | AUPRC | Balanced Acc. | ΔAUROC vs Clinical |
| --- | ---: | ---: | ---: | ---: |
| Clinical-only | 0.7092 | 0.5575 | 0.6611 | – |
| Clinical recalibration | 0.7244 | 0.5842 | 0.6611 | +0.0153 |
| Dynamic MRI-only | 0.5600 | 0.4009 | 0.5347 | -0.1492 |
| Naive Clinical + Dynamic MRI | 0.5890 | 0.4202 | 0.5696 | -0.1202 |
| Clinical + Dynamic MRI Residual | 0.7109 | 0.5646 | 0.6589 | +0.0018 |

Residual vs Clinical：ΔAUROC=+0.001791，95% CI [-0.008330, +0.012029]；ΔAUPRC=+0.007078，95% CI [-0.003973, +0.018827]。

作为 sensitivity analysis，Residual 相对 clinical recalibration 的 ΔAUROC=-0.013478，95% CI [-0.033456, +0.006576]；因此也没有证据表明它超过经过校准的 clinical prediction。

## 时序曲线

| 可用 MRI prefix | Residual AUROC | Residual AUPRC | ΔAUROC vs Clinical | 95% CI |
| --- | ---: | ---: | ---: | --- |
| T0 | 0.7065 | 0.5607 | -0.0027 | [-0.0166, +0.0103] |
| T0→T1 | 0.7109 | 0.5646 | +0.0018 | [-0.0083, +0.0120] |
| T0→T1→T2 | 0.7139 | 0.5642 | +0.0048 | [-0.0033, +0.0132] |
| T0→T1→T2→T3 | 0.7081 | 0.5604 | -0.0011 | [-0.0100, +0.0078] |

T2/T3 是次级、较晚的时间点，不应倒灌到 T0→T1 预测。

## Image reliance

在 T0→T1，real / zero / shuffled trajectory AUROC 分别为 0.7109 / 0.7013 / 0.6983。real 相对 zero 的 ΔAUROC=+0.009650，95% CI [-0.002421, +0.021559]；相对 patient-shuffled trajectory 的 ΔAUROC=+0.012690，95% CI [-0.000979, +0.026322]。两者的 AUROC CI 均跨零，不能把较高的点估计解读为确证的患者特异性时序效应；它们不参与模型选择。

## Correction analysis

clinical confident-but-wrong 组 n=34，平均 |Δl|=0.0791，有益/恶化=22/12。

## 结论与下一步

结论为 **NO CONFIRMED EARLY DYNAMIC MRI VALUE**。若 T0→T1 的 paired CI 不支持正增益，不应以 T2/T3 较晚结果倒推早期临床效用；下一步应先在独立 cohort 或多 seed 中确认最早可用的正向 landmark。
