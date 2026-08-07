# Native reproduction 资产契约

本文定义解除 Shortcut Audit 硬门槛所需的最小资产及核验顺序。它不是结果报告，
也不把当前发现的候选五折副本认定为原生 split。

## 1. 必需资产

每个 fold（0–4）至少需要：

1. **clean CoRe-JEPA checkpoint**：必须能以目标分支的 `CoReJEPA` 严格
   `load_state_dict(..., strict=True)` 恢复，且包含 `model`、`config`、
   `condition`、`response_transform`、`patient_ids`、`n_primary`、`splits`、
   `epoch` 和 `validation`；
2. **冻结的 fold-specific readout**：必须是训练结束后保存、可调用
   `predict_proba` 的对象，不能在 held-out test 上重新拟合；
3. **分类阈值及来源**：给出数值、选择规则和只使用 validation 的证据；若原实现
   实际固定为 0.5，应明确写为 fixed threshold，不能称为 validation-selected；
4. **split provenance**：train/validation/test patient ID、patient order、seed、
   manifest 原路径或内容哈希；
5. **native reference**：T0、T0–T1、T0–T2 的 fold-level prediction 或至少参考
   metrics，以及生成它们的命令、config 和 commit；
6. **预处理 provenance**：DCE8 tensor cache 与 response cache 的生成 config、
   manifest/version；若只提供原始数据，则需确认允许用目标分支的原始脚本重建 cache。

五个 checkpoint 必须各自对应五折中的 train/validation/test，不能用一个单次
70/15/15 checkpoint 重复标成五折，也不能用 legacy schema checkpoint 替代。

## 2. 候选 manifest 的当前身份

当前可访问候选：

```text
<ISPY2_ROOT>/
  _matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/
  matched_patient_cv_splits_seed2026.csv
```

其 SHA256 为：

```text
143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38
```

它覆盖 808 名 clean I-SPY2 患者、label 完全一致，且每名患者恰好一次进入 test。
但它来自缺失的外部 run 目录副本，clean 分支没有引用它。只有当五个 checkpoint
内部的 split patient IDs 与它逐折完全一致后，才能把它升级为 native manifest。

另一个 seed-3072 legacy manifest 的 held-out assignment 与该候选大幅不同，禁止混用。

## 3. 必须通过的核验顺序

正式执行前按以下顺序 fail closed：

1. 目标 branch/commit 与工作树状态；
2. 五折 manifest 的患者集合、label、一人一次 OOF 与内容哈希；
3. 每个 checkpoint 的 clean payload schema 和严格参数恢复；
4. checkpoint 的 808 名 primary patient IDs、I-SPY1 pretraining-only 隔离；
5. checkpoint 内每折 split 与 manifest 的 train/val/test 逐 ID 一致；
6. checkpoint condition vocabulary、年龄统计、response transform 和 config 恢复；
7. readout 文件身份、输入特征维度、阈值来源；
8. native prediction 的患者数、fold、decision point、label、checkpoint 与 threshold 对齐；
9. 在同一 held-out 患者上重算 native，和参考 prediction/metrics 比较；
10. 只有数值在预先说明的容差内一致，才运行 B–F。

任一步失败都必须停止结论性 audit，保存中文诊断和原始错误日志。

## 4. Primary endpoint 必须先澄清

目标分支当前的 primary FLR 只读取：

```text
geometry + clinical/treatment/nominal-time condition
  -> FutureResponseState -> landmark features -> logistic readout
```

它不读取 `visit_state`、`image_prediction` 或其他 MRI latent。因此 C1 MRI-only
replacement 对该 FLR 的概率按架构应严格不变；temporal/donor audit 中 MRI 交换本身
也不会进入该 readout，只有同时交换的 geometry 会进入。

项目方需确认要审计的是：

- 当前提交的 geometry/condition-only primary FLR；或
- 尚未提交、真正读取 MRI trajectory latent 的 fold-specific readout。

二者回答的是不同问题，不能把前者的零 MRI 敏感性解释成模型已被实验证明“不使用
真实 follow-up MRI”。Copy-current latent audit 仍可检查 JEPA transition，但不能据此
替代 primary pCR endpoint 的 MRI 贡献检验。

## 5. 资产提供方式

资产可以保存在任意只读位置；无需复制到 Git 工作树。请同时给出五折目录映射和
文件哈希。审计代码只会读取这些文件，新 predictions、metrics、figures 和日志全部
写入 `shortcut_audit/`，不会覆盖原资产。
