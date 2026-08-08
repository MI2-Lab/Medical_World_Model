# Radiomics Target Screening 实验计划

## 目标

在不使用 pCR、治疗反应标签或其他临床结局的前提下，从 I-SPY2 Multi-feature MRI NACT quantitative measurement table 中筛选一个与 FTV 互补、具有纵向响应性、能由当前 DCE7 lesion-centered 输入合理观察、且不过度依赖 mask/geometry shortcut 的第二个 Direct Grounding target。

## 范围约束

- 本实验仅做表格数据分析、既有证据审计与 target screening。
- 不训练神经网络，不修改 G1/G3、JEPA、transition 或既有实验。
- 不使用 pCR 或任何 downstream clinical outcome 参与候选筛选、排序或阈值选择。
- 所有新产物仅写入 `additional_experiments/radiomics_target_screening/`。

## 数据与证据

- 主表：`/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx`
- 既有证据：
  - `additional_experiments/observed_state_radiomics_audit/`
  - `additional_experiments/direct_grounded_response_state/`
  - `additional_experiments/g3_multiseed_generalization/`
  - `additional_experiments/grounding_jepa_conflict_audit/`
- 正式筛选尽量沿用 seed-2026 patient-level five-fold split；每个 outer fold 的指标仅使用该 fold 的训练患者拟合和计算。

## 分析流程

1. 真实检查所有 Excel sheets、字段、类型、缺失、重复、患者 ID、visit 编码与纵向结构。
2. 将字段分类为患者标识、visit/time、FTV、MRI-derived quantitative candidate、clinical/metadata 与 non-feature；正式候选排除 pCR、分子亚型、治疗、年龄、结局和管理字段。
3. 统计各候选的患者/visit/transition coverage、分布质量和动态范围。
4. 在每个训练 fold 中计算 static FTV redundancy、delta FTV redundancy，以及由 train-fold fitted StandardScaler + Ridge 得到的 residual information。
5. 评估 transition-level standardized change、变化方向、near-zero change 与 within-patient/total variance。
6. 计算 static/delta candidate pairwise Spearman matrices，并审计既有 mask/geometry shortcut 与 frozen representation decodability 证据。
7. 根据当前 DCE7 lesion-centered crop 的实际 input contract 进行 MRI observability gate。
8. 使用 multi-criteria/Pareto-style decision matrix 给出唯一第一候选、第二候选及不推荐原因；探索性标准化分数仅用于可视化，不作为单一决策依据。

## 主要方法约定

- 相关性主指标为 absolute Spearman，Pearson 仅作辅助。
- 变化定义为相邻 visit 的原始差：T0→T1、T1→T2、T2→T3。
- residual 模型保持简单：按 fold/visit 或 fold/transition 在训练患者内拟合 `StandardScaler + Ridge`；screening 前固定并记录简单变换（static FTV/LD=`log1p`，SPH/BPE=identity；raw delta 不变换），不按结果或 fold 自适应挑选。
- robust scale 使用训练数据 IQR；不任意删除 outlier，只报告和标记 heavy tail、floor/ceiling、near-constant 与 extreme-tail。
- 患者匹配只允许既有验证映射或严格 normalization，不做 fuzzy matching。
- shortcut 或 decodability 若无既有证据，明确记为 `UNKNOWN`，不重新训练模型补证据。

## 停止条件

当规定的 CSV、10 类图、`table_schema_report.md`、`final_report.md` 和 `final_target_selection.json` 全部生成并通过完整性核验，且报告明确回答 A–J 科学问题后停止；不运行后续 dual-grounding pilot。
