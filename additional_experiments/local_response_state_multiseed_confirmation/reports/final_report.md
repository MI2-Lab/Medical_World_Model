# LOCAL response-state 五种子确认实验：最终报告

本报告记录 `local_response_state_multiseed_confirmation` 的正式、冻结后结果。实验比较 GAP0、GAP3、LOCAL0、LOCAL3 四个架构，在五个训练种子（2026、3026、4026、5026、6026）和五个外层 fold 上形成 100 个训练单元。每个训练种子先将五个 outer-test fold 的 out-of-fold 预测合并，随后以训练种子作为独立单位；fold 层面只作配对敏感性分析，不被当作独立重复。冻结配置、代码和上游来源由 `PREREGISTRATION_LOCK.json` 锁定。训练、特征导出和探针均未使用测试影像、PCR 或 delta FTV 作为模型输入；FTV 仅处于目标/掩码与事后评价侧。

结果文件中的置信区间是固定 RNG seed 20260811、10,000 次按训练种子重采样的 percentile 95% bootstrap CI。所有阈值都是预注册的决策阈值，而非显著性检验。训练矩阵的父调度进程在最后一个子任务完成前退出，因而没有写出终态清单；在不重跑训练的前提下，项目用冻结调度和 100 个最终 cell 的 selection、checkpoint、history、transform 与初始化证据逐项复核后，恢复了私有终态清单。该恢复仅补足已完成 artifacts 的清单，不改变训练结果。

## 1. LOCAL0 相对 GAP0 是否在五个种子中稳定

是。LOCAL0–GAP0 的 static FTV macro Spearman 在五个种子中的效应依次为 0.275、0.252、0.162、0.208、0.143；五个种子均严格大于 0.10，种子均值为 0.208（95% bootstrap CI 0.164–0.252）。Observed delta FTV macro Spearman 的效应依次为 0.232、0.192、0.172、0.158、0.117，五个种子均为正，均值 0.174（0.140–0.208）。因此稳定性门槛（至少 4/5 个种子满足相应方向与幅度）完全通过；详见 `seed_level_summary.csv` 和 `table4_paired_architecture_effects.csv`。

## 2. static FTV 的改善幅度

以 LOCAL0–GAP0 的 pooled OOF endpoint-macro Spearman 为主指标，平均改善为 0.208，种子间 SD 为 0.056，范围为 0.143–0.275。该结果既超过严格的 0.10 单种子门槛，也超过严格的 0.10 种子均值门槛。`table2_static_ftv.csv` 提供 endpoint 与架构的完整静态 FTV 指标，图 02 展示 Spearman 比较。这里的解释应限于训练出的图像状态表征对肿瘤负荷静态关联的预测性增强，而不是因果临床效应。

## 3. observed delta FTV 的改善幅度

LOCAL0–GAP0 的 observed delta FTV macro Spearman 平均改善为 0.174，95% bootstrap CI 为 0.140–0.208，五个种子均为正。该指标针对相邻时间点的观测 FTV 变化，完整 endpoint 结果位于 `table3_observed_delta_ftv.csv`，图 04 给出 Spearman 比较。该结果支持 LOCAL 状态归纳偏置保留与反应轨迹相关的图像信息；它不表示模型使用了 delta FTV 作为训练输入。

## 4. natural R² 是否出现系统性恶化

没有。LOCAL0–GAP0 的 static FTV macro natural R² 种子效应为 0.094、0.109、0.190、0.090、−0.026，均值 0.091（0.024–0.151）。其中 4/5 为正、仅 1/5 为负，未达到预注册的“系统性恶化”判定；因此该保护性条件通过。这个结论不应被误读为每个种子、每个 endpoint 都提升：6026 种子为小幅负值，正是保留 seed-level 报告和图 03 的原因。

## 5. 预测压缩与校准

`table7_prediction_variance_calibration.csv` 和图 06、图 07 给出 prediction/target variance ratio 与描述性 calibration slope。它们用于检查预测是否被不合理地压缩到窄范围，并作为性能数值的解释背景，而不是新增确认门槛。所有公开表格均是去标识化的汇总，不含患者级预测、患者 ID 或缓存路径；患者级输出保留为私有文件。

## 6. LOCAL3 相对 LOCAL0 的静态与动态表现

LOCAL3–LOCAL0 的 static FTV macro Spearman 在五个种子均为正：0.026、0.041、0.022、0.019、0.040，均值 0.030（0.022–0.038）。其 observed delta FTV 效应为 0.034、0.038、0.041、−0.011、0.059，四个种子为正，均值 0.032（0.009–0.050）。故 static 与 delta 两项“至少 4/5 种子为正”的确认条件均通过。效应幅度应与其较窄的 bootstrap 区间和完整 fold 配对表一起解读，而不是只依据单一汇总数值。

## 7. 优化安全性

LOCAL3–LOCAL0 的优化安全性为 25/25 个严格配对 fold 通过（100%）。每个通过项均要求 primary selection、`experiment_pass=true`，且 selected validation state loss 不超过匹配 LOCAL0 baseline 的 1.05 倍；没有 epsilon 放宽或 fallback 通过。GAP3–GAP0 参考比较同样为 25/25。结果超过预注册的至少 23/25（90%）门槛，详见 `table6_optimization_safety.csv` 与图 09。训练患者顺序的跨臂审计以及特征导出顺序审计均为 PASS。

## 8. 多种子确认结论

正式分类为 `LOCAL_MULTISEED_CONFIRMED`。其依据是：LOCAL0 对 GAP0 的 static 与 delta 条件均在至少 4/5 种子中通过，static 种子均值严格超过 0.10，natural R² 未系统性恶化；LOCAL3 对 LOCAL0 的 static/delta 条件通过，且优化安全性达到 25/25。该结论固定在 `decision_summary.json` 与 `aggregation_summary.json`，并由 Tables 1–7 和 Figures 01–10 支撑。

## 9. 后续架构锁定

根据确认规则，LOCAL 被锁定为下一阶段的图像状态架构。该锁定仅针对本冻结实验定义的空间 LOCAL pooling、相同 encoder stage、投影维度、训练顺序和数据合同；并不自动推广到其他 MRI 序列、其他队列、不同 pooling 几何或不同优化程序。任何架构变更都应建立新的冻结计划，而不能回溯性修改本结果。

## 10. FTV+LD 的授权与范围限制

确认结果授权进入 FTV+LD 阶段，前提是另行冻结其数据合同、模型和分析计划。当前 FTV grounding 只涉及肿瘤负荷/反应信息；它不证明完整 MRI 利用，也不证明超出 Patient Profile 的互补肿瘤表型。尤其，HR/HER2 及其他临床分层不能由本确认实验直接推出，应在独立、预先规定的分析中评估。上述限制防止将表征改善外推为诊断、治疗或生物标志物主张。

## 可复核性与交付状态

本次正式矩阵为 100/100 完成单元，四个架构、五个种子、五个 fold 的笛卡尔积完整。所有选择均为 primary 且通过；公开结果不含患者级 artifacts。冻结锁为 `PREREGISTRATION_LOCK.json`，其 SHA-256 为 `a4e1cd2d8b61a7130da2b2eb6dc04e9a5355f44d0a37f4ceccf2fba48b35a9ee`。正式实现与结果已提交至分支 `feature/local-response-state-multiseed-confirmation`，基线交付提交为 `2d53c51348919a821d1b49c8f5bbb4691b6a641f`，GitHub 推送状态为 `GITHUB_PUSH_SUCCEEDED`。本报告的这条交付状态更新会作为随后独立的文档提交推送，以免在报告中声明尚未存在的自身 commit SHA。
