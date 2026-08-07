# Observed-State Radiomics Audit 验收记录

日期：2026-08-07

## 1. 验收结论

正式产物通过代码、数据覆盖、split 隔离、聚合、图表和 Git 隐私边界检查。没有重新训练或微调 M0/M1/M2 world model；本轮只执行冻结 feature extraction、线性 probe、统计聚合和报告生成。

| Goal 验收项 | 证据 | 状态 |
|---|---|---|
| 1. 不重新训练 M0/M1/M2 | 15 个 formal summary 的 `world_model_trained_or_finetuned=false` | 通过 |
| 2. 使用已有五折 checkpoint | M0/M1/M2 各 fold 0–4，共 15 个 `best.pt` SHA 锁定 | 通过 |
| 3. Patient-level fold 隔离 | canonical manifest/hash、feature patient order 与 split fail-closed；scaler 仅 train、alpha 仅 validation | 通过 |
| 4. Global observed-state probe | projected、pre-projector、GAP，online/EMA 均覆盖 | 通过 |
| 5. Observed global-delta probe | D1/D2/D3/D4，三个相邻 transition 均覆盖 | 通过 |
| 6. ROI/local probe | 128-D ROI occupancy-weighted mean 及 D5/D6 均覆盖 | 通过 |
| 7. 三个主 target | FTV、LD、sphericity 完整；BPE 另作探索 | 通过 |
| 8. M0/M1/M2 公平协议 | 相同 Ridge、alpha 网格、split、target transform、patient mask 和指标 | 通过 |
| 9. Prediction-level 保存 | 15 个 formal CSV，共 743,544 条 outer-test prediction | 通过 |
| 10. 完整中文报告 | `reports/final_report.md` 含要求的 14 个主题和 8 个明确回答 | 通过 |
| 11. 明确瓶颈诊断 | 报告给出 target-dependent 多瓶颈与下一步优先级 | 通过 |

## 2. 正式产物闭合

| 项目 | 正式值 |
|---|---:|
| Frozen checkpoint | 15 |
| Feature manifest 行 | 484,800 |
| Formal probe cell | 9,960 |
| Outer-test prediction 行 | 743,544 |
| Fold-level metric 行 | 9,960 |
| Pooled OOF metric 行 | 1,992 |
| 核心 bootstrap CI 行 | 3,996 |
| Paired fold 行 | 3,675 |
| Paired bootstrap CI 行 | 5,145 |
| 配对患者 | 375 |
| Outer fold | 5 |
| 正式图 | 12 |
| 聚合 input issue | 0 |

Formal probe 的 15 个 summary 每个均为 664 个 cell。Selection key 无重复；`test_used_for_scaler=false`、`test_used_for_alpha_selection=false`，且 `test_predict_call_count=1` 对所有 cell 成立。

## 3. 实现与资产 hash

| 对象 | SHA-256 |
|---|---|
| Fold manifest | `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` |
| Feature manifest | `6a31db77415f8203038ff2defc9a58e77ebdaefe981e3d618d4c00a2f411376b` |
| Feature extractor 实现 | `3065ec5f37f6af8bd057a3c41fbdc0374287fb46a9a5c5bbd2225309aa8713bf` |
| Probe 实现 | `d4b57b395b62cf5f022ad01a18f0ac66d35e78f0f142dc50b442d4480a98ad1d` |
| 聚合/图表实现 | `cd50b2e8fe5cca5b8c75114416429e782640313a13be95665762fc300a219a6e` |
| Canonical raw-radiomics mapping | `512749ccf986de4af4c0109b4ce060c61a90112816895a2ae7423784ea60de4e` |

Formal metadata 中记录的实现 hash 与当前源码重新计算结果一致。五折 static transform 与上一轮五折 change transform 均锁定 train patient hash；`static_endpoint_equality_verified=true`，`test_used_for_fit=false`。

## 4. 执行的工程检查

- `python -m compileall -q src scripts`：通过。
- `python scripts/run_probes.py --self-test`：single-output Ridge、train-only scaler、validation-only alpha、finite output 通过。
- `python scripts/aggregate_results.py --self-test`：59,760 条 synthetic prediction、12 张图、25 次 smoke bootstrap、临时输出回收通过。
- 对已有 `final_analysis` 不带 `--overwrite` 运行：按设计以 `FileExistsError` 拒绝覆盖。
- 正式聚合状态：`complete`；代码 hash 与 summary 一致；input issue 为 0。
- 12 张正式图逐张目视检查：标题、图例、坐标和说明没有遮挡或裁切。
- Markdown 的所有本地相对链接均存在。
- 可提交 aggregate CSV 均不含 `patient_id`、`trial_id` 或 `ispy2_id` 列。

## 5. Bootstrap 与解释边界

核心 CI 使用 2,000 次 patient-within-fold percentile bootstrap；paired CI 只在相同 patient set 上比较。它们条件于已经拟合的 probe，不覆盖训练随机性。D1/D2/D4、EMA、BPE、B1 以及 static B2/B3 只有点估计；B2/B3 仅 change observed-difference 核心 cell 有正式 CI。

Observed delta 看见真实 endpoint MRI，transition-predicted delta 只看当前 prefix；该比较只用于瓶颈定位，不是 information-matched forecasting baseline。输入包含二值 ROI mask，必须把主 representation 称为“ROI 辅助 observed image representation”。

## 6. Git 与隐私边界

- 提交：源码、配置、中文 Markdown、去标识 aggregate metrics、12 张 aggregate 图。
- 不提交：patient-level feature manifest/NPZ、prediction-level CSV、per-cell selection records、运行日志、checkpoint、cache。
- `.gitignore` 明确保留本地 `features/**`、`predictions/**`、`metrics/probes/**` 和 `logs/**`；只放行 `metrics/final_analysis/**`、target transform 验证 JSON 和 `figures/final_analysis/**`。
- 本轮只修改 `additional_experiments/observed_state_radiomics_audit/`；既有 `radiomics_next_change/` 与 `shortcut_audit/` 不在提交范围。

## 7. 正式入口

- [实验计划](../EXPERIMENT_PLAN.md)
- [资产检查](asset_inspection.md)
- [最终报告](final_report.md)
- [聚合摘要](../metrics/final_analysis/aggregation_summary.json)
- [OOF 指标](../metrics/final_analysis/oof_metrics.csv)
- [核心 CI](../metrics/final_analysis/bootstrap_ci.csv)
- [配对 CI](../metrics/final_analysis/paired_differences_bootstrap_ci.csv)
