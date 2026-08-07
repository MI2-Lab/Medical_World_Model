# CoRe-WM Shortcut Audit

本目录只包含审计包装、输入/输出契约、前置校验和诊断记录；不会修改
`ispy_jepa_tmi_clean/` 的核心方法，也不会覆盖原 checkpoint、split 或结果。

## 当前状态

状态：**五折 audit retraining、A–F 审计、患者级 bootstrap、八张必需图和最终中文报告均已完成**。

已经确认目标分支和提交，完成仓库/数据/模型路径检查、候选五折 manifest 校验、
真实单患者 cache smoke、完整 paper 维度 GPU forward，以及审计组件的单元测试。
原始 fold checkpoint 没有随仓库提供；按用户后续授权，所有实验都明确标为基于当前
repo 设计的五折审计重训练，不冒充原论文 checkpoint 或论文数值复现。

当前协议见 `report/retraining_protocol.md`；使用已验证 seed-2026 五折，每折独立训练
representation/readout，held-out test 不参与训练、超参数或 threshold 选择。

五折 checkpoint 均通过 schema、split、patient order、latent 防塌缩和 SHA256 校验；
held-out test 合并后覆盖 808/808 个唯一 I-SPY2 患者。机器可读训练核验见
`metrics/fivefold_training_validation.json`。

正式结果覆盖 native、copy-current、repeated-T0、temporal-order、matched
follow-up swap 和 F1–F5 简化输入基线。完整的逐折与合并 OOF prediction
保留在本地 `predictions/`；为避免向 Git 提交患者级大表，GitHub 分支只发布
去标识化汇总表、图、代码、测试和中文报告。结论包的 SHA manifest 位于
`metrics/conclusion_artifacts_manifest.json`，完整报告见 `report/shortcut_audit_report.md`。

原始资产缺失的历史证据与实现差异见：

- `report/repository_inspection.md`
- `metrics/prerequisite_check.json`
- `metrics/fold_manifest_validation.json`
- `logs/native_reproduction_preflight.log`
- `report/reproduction_asset_contract.md`
- `report/retraining_protocol.md`

## 审计保护层

`auditlib/` 中的模块已用于本次正式审计：

- `folds.py`：五折 patient-level manifest、OOF 唯一性及 label 对齐校验；
- `provenance.py`：clean checkpoint schema、patient order、split 与 readout 校验；
- `runtime.py`：从 checkpoint metadata 恢复 config、condition encoder 和冻结模型；
- `perturbations.py`：Repeated-T0 C1/C2、T1/T2 order swap、matched donor swap，
  并固定 perturbation latent audit 使用未扰动 EMA target；
- `matching.py`：held-out fold 内、完全不使用 outcome 的 baseline-only donor matching；
- `metrics.py`：原 JEPA LayerNorm 距离、copy-current gain、分类指标及配对 bootstrap；
- `contracts.py`：prediction-level CSV 固定 schema、label/threshold/checkpoint 对齐和原子写入。

五折正式产物已通过 schema、OOF 唯一性、label/threshold/checkpoint 对齐、
donor outcome-blind matching、latent 公式、manifest SHA 与完整回归测试。

## 可重跑检查

从仓库根目录运行：

```bash
conda run -n bowen python shortcut_audit/scripts/check_prerequisites.py
conda run -n bowen python shortcut_audit/scripts/validate_fold_manifest.py
conda run -n bowen python -m unittest discover -s shortcut_audit/tests -v
```

`check_prerequisites.py` 检查的是已经确认不存在的“原始正式资产”，因此仍预期以
非零状态退出。它保留为历史 provenance；新的五折审计重训练使用独立协议和输出。

## 结果目录约定

所有审计结果均写入本目录下的 `predictions/`、`metrics/`、
`figures/`、`logs/` 和 `report/`。大体积 `.npz`、`.pt`、`.pth`、`.ckpt`、
`.pkl` 与运行日志已由本目录 `.gitignore` 排除。
