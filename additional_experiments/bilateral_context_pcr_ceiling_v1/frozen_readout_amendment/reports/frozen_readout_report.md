# Frozen bilateral DINO cache readout amendment

## 结论

已使用已经完成的 bilateral DINO cache 做 frozen nested-OOF readout。它显示方向性 pCR 信号，但强度不足以解锁 adapter 或 formal matrix：`BILATERAL_DINO_CACHE_SIGNAL_BUT_NOT_ROBUST`。

| timing | DINO global ΔAUROC vs C+FTV | DINO spatial-pooled ΔAUROC vs C+FTV | DINO global AUROC − radiomics AUROC | DINO spatial AUROC − radiomics AUROC |
|---|---:|---:|---:|---:|
| T0 | 0.0112 | 0.0112 | 0.0094 | 0.0094 |
| T0_T1 | 0.0113 | 0.0092 | 0.0145 | 0.0124 |
| T0_T2 | 0.0129 | 0.0101 | 0.0098 | 0.0070 |

paired DINO-versus-radiomics bootstrap 的 early CI 仍跨越 0；因此不能声称 DINO 已经稳定超过 radiomics。这里的 `DINO_SPATIAL` 是对 4×4 spatial tokens 做 pooled embedding，并不检验二维 patch layout 本身的增量。

## 研究含义

当前结果支持“image 中存在比固定 16-D radiomics target 更丰富的候选信号”，但尚未支持“该信号稳定预测 pCR”。下一步应优先验证 frozen bilateral readout 的稳健性（预先固定的 fresh split/seed 或独立 cohort），而不是继续堆叠 adapter、radiomics target 或 foundation model。
