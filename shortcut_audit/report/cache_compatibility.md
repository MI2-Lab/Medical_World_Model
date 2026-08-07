# 审计重训练影像缓存兼容性报告

## 结论

本次五折重训练使用下列现有 cache 作为独立、明确命名的 canonical 数据版本：

```text
/data/data/Preprocessed/I-SPY2/
  _mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_
  autoroi_t0fallback_minfrac05_z32_y96_x96
```

它适合“参考 repo 模型设计、自行训练”的新实验，但不能声称与目标分支 clean
`build_patient_tensor` 逐位或逐患者完全等价。

## 覆盖与格式

- 覆盖 clean loader 的 964/964 人：808 I-SPY2、156 I-SPY1；无缺失、额外或重复；
- 964 个 aggregate NPZ，每个只有 `x float32 [4,8,32,96,96]`；
- NPZ payload 总计 36,390,028,288 bytes；
- patient ID 清单 SHA256：
  - 全部：`83110e1cf47581a977f63fe6935021031ec9ad72ddba519e673892d011ad5a2c`
  - I-SPY2：`aea53481c12c98aca770281640593ac98ef3eadcbfbb458f2d3a476d7f468b63`
  - I-SPY1：`90a50fd6bec532336a5bd65c537629848f553583a896b0b3c216565587c159d7`
- 文件名、大小、x.npy CRC32 与 header 的 inventory SHA256：
  `fdd1474b9cdbf202bc34b12e1e5b08e180a1f1da141c8f8b2edd5883fabcd658`。

## 与 clean 实现的差异

对 808 名 I-SPY2 的 3,232 个 ROI crop 全量比较：

- 3,078 个 visit ROI 完全相同；
- 154 个 visit 不同，涉及 77/808 人；
- T0/T1/T2/T3 差异数为 26/26/36/66；
- 差异汇总 SHA256：
  `30bd8c52e380a3828d6ef1d8bb7186a64dfd63ea20310a6486a7b45ba188d8e1`。

主要原因是 clean `_project_center` 在半体素和跨形状投影后的临界舍入造成一体素
crop 位移。反例 `ACRIN-6698-641246` 的全图最大绝对差为 10.0、相关系数 0.8698。

相容例 `ACRIN-6698-102212` 和 `ISPY1_1001` 的最大差为
`4.7683716e-7`、相关系数 1.0、ROI 与派生 q 完全一致。会触发
`min_roi_capture=0.5` 的 `ISPY1_1005` 也仅有同量级 ULP 差；不带 `minfrac05` 的旧
cache 则最大差 6.3135，因此不得替换首选目录。

## Audit adapter

`LegacyXCacheDataset` 执行：

1. 只读加载每名患者的 `x`；
2. 验证 `[4,8,32,96,96]`、float32 和 finite；
3. 从第 8 个 ROI channel 调用 clean `mask_geometry` 重算 `q_t [4,9]`；
4. condition 使用每折 train + I-SPY1 拟合的 encoder；
5. 不读取 cache 中的 outcome、旧 geometry 或 split。

Checkpoint provenance 会记录 cache 路径、adapter 模式和本报告 inventory hash。

## Response cache

检查 23,824 个现有 NPZ 后，没有发现覆盖 964 人并符合
`x_visit + patient_ids + feature_names` schema 的 106-D response cache。因此该部分
使用 clean `corejepa/data/response_targets.py` 重新构建到 audit 专用路径：

```text
shortcut_audit/cache/corejepa_response_features.npz
```

正式构建耗时 1,126.42 秒、峰值 RSS 约 5.27 GiB；输出为
`x_visit [964,4,106]`，经 clean `response_vector` 得到 `[964,3,18]`。患者顺序、
106 个 feature 名称及顺序均与 clean loader 完全一致，文件 SHA256 为
`87698b7cd4f7d0130c30a6dac58958948dc094e29f3659f646ee2dd7ea120ac0`。

ROI 来源共 3,856 个 patient-visit：3,166 个 `ftv_inclusion_region`、66 个
`legacy_full_field_empty_ftv`、4 个 `automatic_enhancement_roi`，以及 I-SPY1 的
620 个 `released_longest_diameter_proxy`。缺失 kinetic descriptors 由每折仅在
`pretrain_train` 拟合的 response transform 处理，不使用 pCR。机器可读校验记录见
`metrics/response_cache_validation.json`。
