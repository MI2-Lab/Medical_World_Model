# 最终核验记录

日期：2026-08-06
分支：`feature/ispy-clean-corejepa`
基准 commit：`c413ec86af04795434bdc19e65bbb006c966f379`
环境：conda `bowen`

## 1. 核验结论

最终核验通过。正式结果目录为 `metrics/final/final_analysis_v2/`，状态为 `正式聚合完成_complete`，`input_issues.csv` 只有表头、无问题记录。独立聚合复核未发现 blocker 或 high-risk 问题。

输入契约已统一核实为 8 通道 `DCE7 + binary ROI mask`：模型不读取独立 clinical、treatment、9-D geometry descriptor 或 radiomics，但因 mask 仍携带 ROI 信息，所有主要方法均标记为“ROI辅助 image-only”。

## 2. 实现级检查

- 在 conda `bowen` 中重新执行实现验证，状态为“通过”。
- 808 名 primary I-SPY2、156 名额外 image pretraining I-SPY1、375 名 measurement-paired 患者计数一致。
- 五折 train/validation/test 与各折 radiomics transform 对齐。
- M0、M1 delta-only、M1、M2 的前向、损失和反向均为有限值。
- `lambda_rad=0` 时公共参数更新位级相等。
- `weights_only=True` checkpoint 严格重载通过。
- 18 个 Python 文件全部通过内存语法编译。
- C0/C1/C2 self-test 通过；可观察维度为 T0=4、T0–T1=12、T0–T2=20，未读取未来 measurement，test 未参与模型或阈值选择。

## 3. 产物级检查

对正式 CSV、JSON、checkpoint、图件与报告链接执行了 106 项断言，全部通过：

| 检查项 | 结果 |
|---|---:|
| M0/M1/M2 正式 folds | 各 0–4，共 15 个 best checkpoint |
| 每模型 pooled OOF 患者 | 808，患者×模型×决策点无重复 |
| 原生分类 prediction rows | 7,272 |
| transition prediction rows | 7,272 |
| M2 原生 head rows | 4,500 = 375×3×4 |
| C0/C1/C2 control rows | 3,375 = 375×3×3 |
| post-hoc probe rows | 13,500 |
| paired model-difference rows | 12；每行 808 人、2,000 个有效 bootstrap |
| 核心图 | 9；均可解码且至少 900×600 |
| 聚合 issues | 0 |

另外逐单元重算了 9 个 pooled OOF AUROC 和 `average_precision_score`，与正式汇总表逐浮点精度一致；患者 fold 与 pCR 标签跨模型、跨决策点保持一致；375 名 controls 患者集合与 M2 head 患者集合完全相同；最终报告中的全部相对 Markdown 链接均可解析。

## 4. 统计重现边界

- 正式置信区间使用 2,000 次 patient-level percentile bootstrap，master seed 为 `20260806`。
- `AUPRC` 字段采用 scikit-learn `average_precision_score`，即 AP 定义，不是梯形积分得到的 PR 曲线面积。
- Paired AUROC/AUPRC 差值在固定表格顺序中共享同一确定性 RNG 流；整次聚合可重现，但单行 seed 不能脱离该顺序独立重放第二个指标。这不改变置信区间的有效性，也不影响模型选择，因为聚合与 bootstrap 均发生在全部训练和选择完成之后。
- 严格聚合实现 SHA-256 为 `ca8c246483788db58b00c11f81144fcb86d5eb1d83c3492d7865df4f47182562`；同标签重跑会拒绝覆盖，已有正式产物未被改写。

## 5. 保留项

原始数据、共享 cache、候选五折 manifest、原始 checkpoint、旧版 `final_analysis/` 和用户既有的 `shortcut_audit/` 均未删除、移动或覆盖。工作区原始 tracked commit 保持不变；本研究所有新增内容都位于 `additional_experiments/radiomics_next_change/`。
