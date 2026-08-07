# M3 relational loss 门控决定

## 结论

M3 状态为：**按预注册门控未运行**。阻断项不是数值稳定性，而是 M2 radiomics grounding 不足；继续加入 relational loss 会把接近常数、缺少患者级排序能力的辅助预测放大为患者间关系监督，科学风险高于潜在收益。

## 逐项门控

| 预注册条件 | 结果 | 判定 |
|---|---|---|
| M2 无 NaN/Inf、梯度稳定 | 五折均完成；best epoch 为 12/12/12/12/11；所有 latent eligibility 通过 | 通过 |
| validation `L_rad` 优于 fold-train mean predictor | fold 0、λ=0.05：standardized SmoothL1 0.5215；fold-train mean predictor 0.5534 | 通过，但幅度有限 |
| 至少一个 feature 在 validation 显示非零且非近常数的可预测性 | λ=0.05 按 feature 合并三个 transition 后 Spearman 为 FTV 0.067、sphericity -0.024、LD 0.049、BPE -0.018。单个 T1→T2 LD 点估计虽为 0.244，但预测/目标方差比仅 0.0007，属于近常数排序波动，不满足可靠 grounding | **未通过** |
| M2 的 ROI辅助 image-only AUROC 相对 M1 未下降超过 0.02 | 完整 OOF 的 T0/T0–T1/T0–T2 差值为 -0.0081/-0.0094/+0.0001 | 通过 |
| 无 ID、时间点、fold 或 transform 泄漏 | 严格 ID、锁定 manifest、fold-train-only transform；v3 selector 重算 validation gate 并通过独立复审 | 通过 |

正式 OOF grounding 进一步支持停止：12 个 transition×feature 单元中 11 个被标记为 near-constant prediction；Spearman 范围为 -0.091～0.053，只有 T1→T2 BPE 和 T0→T1 LD 的 RMSE 相对 fold-train mean baseline 有极小正 gain（约 0.45% 和 0.07%），其余均不优于均值预测。

## 后果

- 不运行 `lambda_rel` 搜索，也不使用 pCR 构造关系 pair。
- 不把 C0 radiomics-only 或 C1 direct fusion 的较高 AUROC 当作 M2 成功；二者推理时使用 measurement，不符合主要 image-only 问题。
- 下一轮若继续，应先改进可预测的局部/功能性影像目标、去除 ROI mask 重复性，并在 validation 上建立非近常数 grounding，再重新开启 M3 门控。
