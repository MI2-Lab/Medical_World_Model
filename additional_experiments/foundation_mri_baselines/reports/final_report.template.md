# DCE-MRI Foundation Encoder Baselines：正式公开结果报告

本报告由冻结的 public-only finalizer 从公开 aggregate CSV/JSON 一次性生成。生成器不接受 private prediction、patient identifier、模型筛选、metric 排序或 top-k 参数；所有数值均按预注册 identity 字典序完整呈现。AUROC/AUPRC/Brier 配对差值统一为“候选减参照”。区间为固定患者级 bootstrap 的描述性 95% percentile interval，不作显著性检验、等效性判断或模型选择。

## 1. 全覆盖验收

{{COVERAGE_MANIFEST}}

正式 foundation 集合固定为 MedicalNet 3DSeg-8 ResNet-50 与 DINO v1 ViT-B/16。Current-CNN 配套参照固定为 `GAP0@GLOBAL` 与 `LOCAL0@LOCAL`。报告正文和附录均不得因结果方向删除任何 model、spatial axis、timing、population 或 endpoint。

## 2. 模型选择与完整 provenance

{{MODEL_PROVENANCE}}

选择边界详见 [foundation model selection](foundation_model_selection.md)；完整执行状态见 [model execution ledger](model_execution_ledger.md)；current-CNN 来源见 [current CNN provenance audit](current_cnn_provenance_audit.md)。

## 3. 十二问

{{QUESTION_MATRIX}}

## 4. 固定规则 scientific diagnosis

固定 cell 规则为：AUROC 与 AUPRC 差值均大于 0 且 Brier 差值不大于 0 才记作 favorable；二者均小于 0 且 Brier 不小于 0 才记作 adverse；其余全部记作 mixed。只有预定集合中的每个 cell 均 favorable 才写“方向一致支持”。

{{FIXED_CONCLUSIONS}}

## 5. 全部 pCR pooled OOF cells

本表完整保留 808 人 primary 与 375 人 complete-case sensitivity。`FTV representation / readout` 的左侧说明 representation 是否接受 FTV supervision，右侧说明 pCR readout 是否把 FTV 作为 covariate；两者不得混写。

{{PCR_TABLE}}

## 6. 全部预注册 paired pCR comparisons

每个 comparison 恰好同时展示 ΔAUROC、ΔAUPRC 与 ΔBrier；绝对 pCR 分数没有 bootstrap interval，不得把本表的 paired interval 移贴到绝对分数。

{{PAIRED_TABLE}}

## 7. HR/HER2 phenotype probes

{{PHENOTYPE_TABLE}}

## 8. 四分类 HR/HER2 subtype probe

Subtype 使用 macro one-vs-rest AUROC/AUPRC、multiclass Brier 与 top-label ECE；不能把 binary calibration slope/intercept 套用于本表。

{{SUBTYPE_TABLE}}

## 9. FTV 与 literal ΔFTV decodability

本表只在同一 375 complete cases 上解释。FTV 可解码不能单独证明 pCR signal 仅由 tumor size 驱动；tumor-size 以外信息的判断由完整 beyond-FTV paired 集合决定。

{{FTV_TABLE}}

完整自动汇总另见 [results summary](results_summary.md)，timing 与 calibration 图分别见 [pCR timing figure](../figures/pcr_timing_performance.png) 与 [calibration/complementarity figure](../figures/calibration_clinical_complementarity.png)。

## 10. Mixed producer lineage

{{PRODUCER_LINEAGE}}

## 11. 失败与执行谱系边界

- v1 baseline 与 probe 均以 exit code 143 终止，未产生可复用或被查看的正式 prediction/metric artifacts；但已在内存处理部分 outcome/test prediction，因此不能写成“未运行 test”。
- 初版 multinomial-SAGA outcome-blind runtime smoke 在预声明边界以 exit code 143 终止；最终显式四分类 one-vs-rest liblinear smoke 通过。
- probe-v2 正式运行以 exit code 1 失败，没有写出 prediction、selection 或 public metric artifact。原 traceback 未保留；metric-free 日志窗口与完全 synthetic、outcome-blind 复现高度匹配于 static-FTV transformed Ridge prediction 经 `expm1` 溢出后的 nonfinite outer-test prediction。这是最可能根因，不应写成已恢复原 traceback的确定事实。
- 最终发布必须在 execution ledger 中同时保留 v1、v2 与最终成功版本的 lock、receipt、artifact SHA 和真实状态；不得用最终成功记录覆盖历史失败。

## 12. 限制

1. 本轮是 frozen encoder 加线性/Ridge probe，不是 fine-tuning 上限。
2. 没有外部 cohort；bootstrap 仅描述本 cohort 的 OOF prediction 不确定性。
3. 375 人 FTV/radiomics complete-case 缺失非随机，绝对指标不得与 808 人直接排名。
4. LOCAL 继承冻结的 T0 localisation prior，不能描述为完全无定位先验。
5. DINO 是固定 2-D axial adaptation，不代表原生 3-D architecture。
6. MedicalNet 上游未发布 checksum；可重复性属于取得时 hash 与 strict-load 的 conditional pass。
7. 所有多模型、多时点区间均为未作 multiplicity adjustment 的描述性结果。
8. 区间跨 0 不证明严格等效；区间未跨 0 也不作确认性显著性声明。
9. Report 与 coverage receipt 分别跨目录发布，不能在 SIGKILL 下构成 filesystem transaction；coverage receipt 是唯一 publication commit marker。若只存在其中一个，下一次运行 fail closed 并要求 operator audit，不自动恢复。
10. AUROC/AUPRC/Brier/ECE 与定义上必有值的误差指标必须 finite 并满足固定域；constant prediction、constant target 或 `b0_rmse=0` 时定义上不可用的 calibration、correlation 或 RMSE gain 以 `NA` 原样保留。任何 infinity 均拒绝，JSON `null` 必须与同 identity 的 CSV `NaN` 对齐。

## 13. 公开输入 provenance

{{PUBLIC_PROVENANCE}}

公开 coverage receipt 另行记录 canonical identity digests、question subset counts、固定结论分支和本报告 SHA。Patient-level features、predictions、selection rows、checkpoints、clinical/radiomics source data 与运行日志的 bytes/content/rows 不进入本报告或 Git；mixed-lineage 表只保留 generic artifact role 与不可逆 SHA-256。

## 14. Git handoff

{{GIT_HANDOFF}}

上表只记录首次 substantive content push。报告与 coverage receipt 随后的 metadata push 状态不能自引用写回 receipt-bound report；无论第二次 push 成功或失败，都只能在最终交付消息中准确报告，失败后不得再次修改本报告来追写状态。
