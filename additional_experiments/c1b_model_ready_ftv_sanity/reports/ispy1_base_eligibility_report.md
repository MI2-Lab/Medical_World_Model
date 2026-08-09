# I-SPY1 严格、outcome-free 的 Stage A base eligibility 报告

生成日期：2026-08-08
契约版本：`strict_outcome_free_ispy1_base_eligibility_v1`

## 结论

Stage A 对**受限合格集**可继续：156 位四访视患者中，140 位的 T0、T1、T2、T3 均通过；其余 16 位必须 fail closed，不进入 Stage A。不能把该结论外推为全体 156 位患者均可用。

全量结果为 624 个访视中 604 PASS、20 FAIL。所有 20 个失败均没有唯一且满足严格契约的同 study 替代 acquisition。患者资格是严格的四访视交集，不允许用较少访视或降低几何阈值补入。

## 冻结的 outcome-free 契约

Runner 只读取既有 imaging manifest 中的 raw study/series 引用和原始 DICOM；不读取 clinical、pathology、response、FTV、phase-metadata 或其他 outcome 表。

每个候选 acquisition 必须同时满足：

1. `Modality=MR`，且每张图像的 `ImageType[0:2]` 精确为 `ORIGINAL, PRIMARY`。
2. `BodyPartExamined=BREAST`，并拒绝 pelvic/body 矛盾信息。
3. 拒绝 DERIVED、SECONDARY、SUB、MIP/MPR/projection、BOLD、SER/PE map、T2、DWI、scout/localizer、FIESTA、survey/B0 map、segmentation、post-needle 和 label/control。
4. 必须有 T1-DCE 正证据，例如 dynamic/DYN、FL3D/FL_3D、3DFGRE、IR-SPGR/FSPGR/FMPSPGR/SPGR、3D-SAG、PRE/POST、PASS、contrast、CE/T1，或 DICOM `SequenceName` 的 `fl3*`/`fgre3d`。历史上裸 `3D` 仅在 gradient-echo 序列证据成立时接受；计数本身从不作为 DCE 证据。
5. Rows/Columns、IOP、PixelSpacing 必须一致；IOP 必须正交；slice clustering 与 spacing tolerance 均为 0.001 mm，in-plane IPP drift 不超过 0.1 mm。
6. Native dynamic 必须形成无重复、无缺失的精确 `slice × time` 网格，且 `2 ≤ T ≤ 20`。时间身份只来自完整 DICOM temporal key，或经 IPP 验证的 InstanceNumber phase blocks；多个完整 key 必须一致。
7. Phase stack 的每个组件必须是完整 3-D volume。默认要求共同物理 frame；不同 frame 只有在方向差不超过 1°、spacing 相对误差不超过 2%、reference-grid 的有效 source footprint 不低于 99% 时才允许线性重采样，并记录插值。该极小外推边界固定用与 C1B 一致的 `reflect`，禁止 constant-zero sentinel；有效覆盖率仍写入 private contract。
8. 当前选择明显错误时，只在同一 study 搜索满足相同语义、时序和几何约束的 acquisition。Phase-run family 使用 laterality、SequenceName、ScanningSequence、SequenceVariant、MRAcquisitionType、TR、TE、FlipAngle 和 EchoTrainLength 的固定 fingerprint。只接受**恰好一个**通过的 acquisition；0 个保持 blocked，多个标为 ambiguous 并 fail closed，不按文件数或分辨率打破平局。
9. 对通过的 source cell，第一次从 raw PixelData 解码并应用 slope/intercept；只有 tag 缺失时才使用 DICOM 默认 slope=1/intercept=0，显式 slope=0 或任何非有限 scaling 必须失败。第二次重新读取、逐 cell 精确比较并复核 hash。然后写入 float32 4-D NIfTI，设置 qform/sform code 1，再读回比较 data、affine 和文件 hash。

Sanitized phase contract 完全由每访视 DICOM 和固定索引规则生成：

- `T ≤ 4`：`pre=0, early=1, late=T-1`
- `T > 4`：`pre=0, early=2, late=min(5,T-1)`

603 个通过访视有可用的 DICOM acquisition/content time；1 个通过访视只有经验证的 phase index，因此只发布索引、不伪造秒数。

## 全量结果

| 指标 | 数值 |
|---|---:|
| 患者总数 | 156 |
| 四访视全部通过 | 140 |
| 至少一个访视失败 | 16 |
| 访视总数 | 624 |
| PASS / FAIL | 604 / 20 |
| 两次精确验证的 raw PixelData cells | 115,906 |
| 同 study 唯一替代 acquisition | 8 |
| 从污染 stack 中保留 ORIGINAL/PRIMARY phases | 20 |
| 安全重采样访视 / phases | 1 / 2 |

访视通过数：T0 150/156，T1 149/156，T2 153/156，T3 152/156。失败患者中，13 位缺 1 个访视、2 位缺 2 个访视、1 位缺 3 个访视。

通过数据的 phase-count 分布为：T2=12、T3=511、T4=35、T5=1、T6=44、T19=1。T19 是结构完整的原始 dynamic acquisition；T80/T85 的 BOLD 选择未被计数规则误收。

独立 header audit 对已知 50 个 hard-block 访视找到 29 个严格修复：7 个 BOLD 的其他 native-dynamic acquisition、20 个污染 stack 的 ORIGINAL/PRIMARY 子集、1 个其他同-study phase run，以及 1 个将错误 50×6 NIfTI 按 raw 100×3 网格重建的 same-series repair。本 runner 另对 1 个访视的 2 个 phase 执行了满足冻结安全阈值的显式重采样（最小有效覆盖率 0.99609375，边界模式 `reflect`），因此最终保留 604 个访视。全 624 个访视按严格 scaling 规则重跑后没有 `INVALID_RESCALE`；失败原因仍唯一为 `NO_CLEAR_SAME_STUDY_REPLACEMENT`。

20 个 blocked 访视的当前 source 根因聚合为：

- `UNSAFE_PHASE_FOV_OVERLAP`: 14
- `UNSAFE_PHASE_GEOMETRY`: 4
- `CURRENT_SEMANTIC_REJECTION`: 1
- `INSUFFICIENT_VALID_PHASES`: 1

这些访视均未找到唯一严格替代 acquisition；没有 ambiguous replacement 被接受。

## 五案例 smoke test

Smoke case 由 runner 按规则确定，不在代码或 public output 中硬编码患者标识。

| Stratum | 预期行为 | 结果 |
|---|---|---|
| clean native dynamic | raw rebuild | PASS |
| clean phase stack | common-frame stack | PASS |
| prior NIfTI 50×6 / raw grid 100×3 | InstanceNumber-block rebuild | PASS |
| selected BOLD | 唯一同-study breast DCE replacement | PASS |
| derived/subtraction，无严格 replacement | fail closed | FAIL |

Smoke 总计 4 PASS、1 预期 FAIL；928 个 raw cells 完成两次精确比较。

## 产物与隐私边界

- Runner：`scripts/run_ispy1_base_eligibility.py`
- Canonical public aggregate：`metrics/ispy1_base_eligibility_summary.json`
- Canonical public failure aggregate：`metrics/ispy1_base_eligibility_failure_reasons.csv`
- Private visit/patient manifests：`manifests/ispy1_base_eligibility_{visits,patients}.private.csv`
- Private sanitized phase contract：`manifests/ispy1_base_eligibility_phase_contract.private.csv`
- Private per-cell provenance：`manifests/ispy1_base_eligibility_cells.private/`
- Rebuilt private cache：`cache/ispy1_validated_dce/`

Smoke 的 aggregate 结果已固化在本报告；临时 smoke public files 在终验后删除，避免与 canonical full-cohort metrics 混淆。重新执行 `--smoke` 会在隔离的 test-log 目录重建它们。

Public aggregate 和本报告不包含患者标识、DICOM UID、source path 或 rebuilt path。Private CSV/JSON 权限为 `0600`。Canonical cache 恰有 604 个 NIfTI，与 604 个 PASS manifest 行一一对应；per-cell audit 也恰有 604 个文件。604/604 个 NIfTI 的重新计算 SHA-256 与 private manifest 完全一致。

## 可复现命令与验证

在 experiment 所在 repository root 运行：

```bash
python -m py_compile \
  additional_experiments/c1b_model_ready_ftv_sanity/scripts/run_ispy1_base_eligibility.py

python \
  additional_experiments/c1b_model_ready_ftv_sanity/scripts/run_ispy1_base_eligibility.py \
  --smoke --workers 8 --progress-every 1 --overwrite

python \
  additional_experiments/c1b_model_ready_ftv_sanity/scripts/run_ispy1_base_eligibility.py \
  --workers 8 --progress-every 25 --overwrite
```

Batch stdout 只报告 aggregate progress，不打印患者标识或路径。`--overwrite` 在重新审计前只删除精确的访视目标，避免失败访视留下旧的 canonical NIfTI。

本报告所述 I-SPY1 source 审计不改变 shared registration、DCE7 channel 公式或模型结构；其 rebuilt NIfTI 只在随后独立的 C1B builder/cache 硬门通过后才可成为 model input。
