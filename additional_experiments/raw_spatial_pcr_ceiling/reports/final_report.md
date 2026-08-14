# Goal C：Raw-Image / Spatial pCR Ceiling Audit

## 结论边界

本报告只估计当前 C1B-H DCE7 输入合同下的**经验性有监督上限**，不是信息论上限、pCR-free World Model、治疗因果证据、外部验证或生产模型。

## 当前运行状态

- 运行状态：`NOT_RUN`
- 预注册决策类：`NOT_RUN`
- 详细机器可读判定：[`../metrics/decision.json`](../metrics/decision.json)

若状态为 `NOT_RUN`，不能把先前实验数值当作本实验的 C1–C5 结果；C0 只能作为历史参考。

## 前置证据摘要

此前已确认 LOCAL3 是稳定的 pooled response baseline；条件 pCR 监督适配的最佳 MRI-only AUROC 约为 0.548，full-encoder fine-tuning 未显著提高；Goal 5 的诊断提示空间局部化信息，但固定 center/inner/outer shells 未恢复稳定 pCR 信号；既有 MRI–clinical 与 foundation audits 也未建立临床或 FTV 之外的稳定增量。因此本实验测试的是“监督空间学习是否能利用 pooling 丢失的信息”。

## 必须回答的问题

1. 空间监督学习是否超过 C0 pooled ceiling？**尚未确定**；当前仅有历史 C0 参考（最佳先前监督 MRI-only 约 0.548），本实验 C1–C5 尚未运行。
2. LOCAL attention 是否优于 LOCAL mean pooling？**尚未确定**；C2 对 C0 的比较待正式矩阵。
3. Patch-token Transformer 是否优于 attention pooling？**尚未确定**；C3 对 C2 的比较待正式矩阵。
4. Full C1B context 是否超过 64-mm LOCAL？**尚未确定**；C4 对 C3 的比较待正式矩阵。
5. Direct raw-image supervised training 是否超过 representation training？**尚未确定**；C5 对 C0 的比较待正式矩阵。
6. 最佳 MRI-only AUROC 多高？**尚未确定**；结果写入 `metrics/mri_only_metrics.csv`。
7. train–OOF 泛化差距多大？**尚未确定**；结果写入 `metrics/generalization_gap.csv`。
8. MRI 是否增加 clinical？**尚未确定**；必须由 `C+M−C` 的 fold-safe 融合填写。
9. MRI 是否增加 clinical+FTV？**尚未确定**；必须在 `ftv_complete_375` 上由 `C+F+M−(C+F)` 填写。
10. 主要瓶颈是 pooling、representation、input 还是 generalization？**尚未确定**；只能由 Gates A–E 与泛化审计共同决定。
11. 是否支持继续 Patch-Token World Model？**尚未确定**；不能用前置实验或漂亮 attention map 代替 Gate A。
12. 后续应 LOCAL-only 还是加入 broader context？**尚未确定**；Gate C 运行后再按 preregistration 解释。

正式运行后，以上答案必须由 arm×timing MRI-only 表、train/validation/test gap 表、clinical complementarity 表、beyond-FTV 表、attention concentration diagnostics、LOCAL vs full-context 表、paired bootstrap 表和 seed consistency 表更新；不允许用外层 test 进行架构选择。

## 公开产物

- `metrics/mri_only_metrics.csv`
- `metrics/generalization_gap.csv`
- `metrics/paired_bootstrap.csv`
- `metrics/decision.json`

患者级预测、原始 MRI、pCR 标签、临床表、特征与 checkpoint 必须留在私有 gitignored 路径。
