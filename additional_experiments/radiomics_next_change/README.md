# Radiomics Next-Change 实验

本目录是在 CoRe-WM 中研究纵向 MRI Next-Change 与训练期 radiomics privileged supervision 的隔离实验实现。正式实验使用五折 OOF 评估 M0 Next-State、M1 Next-Change 和 M2 Next-Change + radiomics auxiliary loss；M2 的正式推理和 pCR readout 不读取 radiomics 表格。由于 MRI 输入的第 8 通道为二值 ROI mask，报告统一称为“ROI辅助 image-only”。

## 主要结论

- M1 将 normalized aggregate transition gain 从 M0 的 -62.21% 提高至 +6.24%，但 raw latent gain 仍为 -2.84%，pCR readout 没有一致改善。
- M2（`lambda_rad=0.05`）与 M1 的 transition 表现近似，且没有改善 pCR；其原生 radiomics head 基本退化为近常数预测。
- 结果不支持“当前低维 radiomics auxiliary supervision 改善 ROI辅助 image-only 治疗响应表征”的主张。

完整解释、限制和 a–j 问题回答见 [最终报告](reports/final_report.md)，实验设计见 [实验计划](EXPERIMENT_PLAN.md)，复核范围见 [最终核验记录](reports/verification_report.md)。

## 公开内容

- `configs/`：M0/M1/M2 配置和五折 train-only radiomics transform。
- `src/rnc/`、`scripts/`：训练、评估、控制组和严格聚合实现。
- `reports/`：全部中文实验报告。
- `data_audit/`：schema、missingness、transition counts 和 complete-case 汇总。
- `metrics/final/final_analysis_v2/`：不含患者标识的最终聚合指标。
- `figures/final/final_analysis_v2/`：九张最终图。

## 未提交的本地产物

为避免向公开仓库发布 patient-level trial ID、pCR 标签和 measurement target，并控制仓库体积，下列内容由 `.gitignore` 保持在本地：

- patient-level overlap 与 raw transition target CSV；
- checkpoint、训练日志和 prediction-level CSV；
- 含患者级行的 transition、shortcut、control、probe 与 OOF 聚合文件；
- smoke、pilot 和旧版可重建指标。

这些文件未被删除。拥有授权数据的研究者可用本目录脚本在相同 fold manifest 和配置下重新生成；公开聚合表保留了样本数、方法指标和 bootstrap 结果，完整方法契约与实现哈希记录在中文报告中。
