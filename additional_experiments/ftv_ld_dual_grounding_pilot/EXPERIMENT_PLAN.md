# FTV + LD 双 Grounding Pilot 预注册实验计划

## 1. 状态与不可变边界

本计划在读取任何 Stage A 汇总结果、启动任何 Stage B 训练前冻结。冻结时间为 `2026-08-08T09:02:18Z`，执行分支为 `feature/ftv-ld-dual-grounding-pilot`，source commit 为 `91ce7e5a26ef3674c56e56e00fe2efa76fdb841b`。该 commit 已与远端 `feature/ispy-clean-corejepa` 核对一致。

环境固定为 conda `bowen`、Python 3.11.14、PyTorch 2.9.1+cu130、CUDA runtime 13.0；可见硬件为 3 张 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，每张约 96 GiB。数据根由 `${DGRS_DATA_ROOT}` 与 `${ISPY2_RAW_ROOT}` 注入，公开文件不保存本机绝对路径。

工作树开始时唯一既有变化是未跟踪 `shortcut_audit/`。它属于用户，本实验不读取、不修改、不删除、不加入提交。所有新增内容只写入本目录。

本任务只有两个问题：

1. 当前 DCE7 lesion/ROI-centered crop 是否足以观察 LD 所需的病灶空间范围？
2. 仅在答案通过预注册 gate 时，FTV+LD 是否比 FTV-only 构建更丰富的 longitudinal response state？

Stage A 为 outcome-free、无训练 audit。Stage A 判为 `NO_GO` 时，立即停止 Stage B，输出 `LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP`，完成报告、验证、commit 与 push；不得降低 gate 以继续训练。

## 2. 既有 screening 证据

已完整读取 `radiomics_target_screening` 的最终报告与 selection JSON。结论按原有限定接受，不重跑 screening：

- LD 是 conditional/pragmatic first candidate，不是统计唯一优胜者；LD 与 SPH 都是 Pareto non-dominated。
- SPH 是第二选择，但具有 HIGH mask-geometry shortcut risk，历史 strict-DCE7 ΔSPH 几乎不可解码。
- BPE 的 static FTV 互补性最强，但其定义需要对侧乳腺中央组织，和当前 lesion-centered input 明确不匹配。
- LD 与 FTV 的 static median |Spearman| 为 0.596，属中等冗余；ΔLD 与 ΔFTV 为 0.369，不是完全冗余。
- LD delta residual variance ratio 约 0.921–0.926，保留较多 longitudinal residual information。
- strict-DCE7 G3 的 static LD / ΔLD macro Spearman 为 0.324 / 0.138，说明弱但非零的可解码信号；对应 R² 为 -0.069 / 0.001，不能夸大。
- 在本轮严格 375 人 overlap 中，LD=0 比例为 T0 0%、T1 1.33%、T2 16.53%、T3 32.53%。T2/T3 有明显 floor。
- crop containment 此前从未逐例验证，旧 observability 只是 conditional architecture gate。

同源论文说明 LD 由 site radiologist 测量并由 study coordinator 从临床 MRI 报告抽取，四访为 T0 pre-NAC、T1 early NAC、T2 inter-regimen、T3 pre-surgery。来源：[npj Breast Cancer 方法](https://www.nature.com/articles/s41523-020-00203-7)。

## 3. LD target semantics

工作簿、同源论文、补充材料和随附 DICOM 文档均未明示当前 `LD_T0`–`LD_T3` 数值单位。0 是真实数值而不是缺失，但来源不能区分 complete response、non-measurable、below detection 或 encoding floor。故预注册：

- 单位状态：`LD_UNIT_NOT_EXPLICIT`；
- zero 状态：`AMBIGUOUS_ZERO_SEMANTICS`；
- 不把工作簿 LD 擅自解释为 cm 或 mm；
- 可以做不受线性单位换算影响的 Spearman rank sanity check，但不做 reported LD 与 segmentation-derived mm 的数值差、ratio 或校准误差解释；
- T2/T3 0 值不删除、不填假值；若进入 Stage B，`log1p(0)=0`。

字段映射固定为 `LD_T0/T1/T2/T3 → T0/T1/T2/T3`。

## 4. Cohort 与真实 input contract

严格 cohort 是锁定 808 人 complete-four-visit MRI cohort 与 384 人 measurement workbook 的精确六位 clinical-trial subject ID 交集：375 人、1,500 个 patient×visit。不得 fuzzy match，不得加入 workbook-only 患者。

当前 legacy cache 每人 `x` 为 `[4,8,32,96,96]`；DCE7 是前 7 通道，第 8 通道是只用于 preprocessing/审计的 binary localization support，模型 B0/B1/B2 均不得读取它。Stage A 必须从 source commit 的真实 preprocessing 实现和 full-resolution NIfTI 重建：

- NIfTI 先规范为 `[X,Y,Z]`；model tensor 输出为 `[Z,Y,X]`；
- 固定 crop 为 `(Z,Y,X)=(32,96,96)` voxel，不做 crop 前 spacing harmonization；
- I-SPY2 以 released T0 bbox center 为中心，并按 source/target shape 的 normalized coordinates 投影到 T1–T3；
- automatic ROI 仅在 projected crop 捕获少于 32 voxel 或少于 50% support 时 recenter；
- source volume 外零填充，source 内按整数 round center 裁剪；
- full support 优先为 FTV inclusion region；它来自 bit-encoded inverse analysis mask，不是手工 dense lesion segmentation。必要 fallback 必须逐例记录，不能混称真实 lesion mask。

legacy cache 没有保存 crop origin，且其历史 builder source 已缺失；现行 clean center 公式只作为候选，不能被当作所有 legacy crop 的逐体素真值。对每个 visit，先围绕 clean start 在 XYZ 各 ±2 voxel 内搜索，把 full mask 直接 crop/pad/XYZ→ZYX 后与真实 cache 第 8 通道逐体素比较；唯一 zero-mismatch candidate 才能用于分侧 margin。多候选取距 clean start 最近者并标记 ambiguous；无候选则把 origin/margin 标 unavailable，不伪造。exact origin recovery fraction 低于 99% 时不得给 GO。

Containment ratio 的主计算不依赖 origin：full mask 1,500/1,500 均非空，binary cache 没有 spatial resize，因此使用 `cached_support_voxels/full_support_voxels`。cache mask 为空而 full mask 非空时 ratio=0、`complete_miss=true`、`severe_truncation=true`，不能因六面均未触碰而判安全。

## 5. Stage A 指标

每个 patient×visit 的本地受控表保存：full-support bbox、requested/effective crop bbox、六个 face touch、`any_boundary_touch`、三轴 voxel/mm extent、crop physical extent、三轴和全局最小 signed margin、full/crop support voxel count、containment ratio、support provenance、spacing 可靠性、reported LD 原始值、LD zero flag、近似 3-D maximum Feret extent，以及 cache reconstruction 状态。

定义冻结于 `configs/stage_a.json`：

- `suspected_truncation`：任一 boundary touch，或 exact containment ratio <0.99，或 diagnostic support unavailable；
- `severe_truncation`：exact containment ratio <0.90；
- `sufficient_containment`：support 可诊断、ratio ≥0.99、且无 boundary touch；
- signed margin <0 表示 full support 已超出 requested crop；
- `approx_max_extent_mm` 由 full support 的固定方向 extrema 候选两两最大距离近似，只作 spatial sanity check，不替代 radiologist LD target。

按 T0/T1/T2/T3 分别报告 n、可审计率、boundary-touch、suspected/severe truncation、median 和第 5 百分位 margin、containment ratio、LD zero floor。Large lesion 使用 outcome-free 375 人中每个 visit 的 descriptive top 25% 与 top 10% LD；同时报告 pooled T0/T1 `Spearman(reported LD, minimum margin)`、contained/truncated LD 分布。所有 patient IDs 只存在被 `.gitignore` 排除的本地表；公开表只含 aggregate counts/rates。

## 6. Stage A GO / NO-GO gate

以下阈值在看结果前冻结：

1. T0/T1 合并 suspected truncation rate ≤10%；
2. T0/T1 top-quartile LD suspected truncation rate ≤20%；
3. 所有 visit 合并 sufficient containment rate ≥85%；
4. T0/T1 pooled `Spearman(LD, margin)` 必须大于 -0.40，避免明显“LD 越大、margin 越差”的系统 mismatch；
5. actual legacy crop 的 exact origin recovery fraction ≥99%。

`GO` 要求五项全部满足且 exact/diagnostic support 完整。`GO_WITH_CAVEAT` 只允许 exact support 部分缺失但 T0/T1 diagnostic support fraction ≥90%，并且把不可审计行保守计为 suspected truncation 后仍满足上述全部 rate gate。其他情况均为 `NO_GO`。

不因 T2/T3 floor、算力或进入 Stage B 的意愿降低阈值。Stage A 的 primary gate 以 T0/T1 为主；combined ≥85% 是独立 safety gate。

## 7. Stage A 产物

本地受控：

- `metrics/crop_containment_patient_visit.csv`（含 patient ID，禁止提交）。

公开：

- `metrics/crop_containment_summary.csv`；
- `metrics/crop_containment_by_timepoint.csv`；
- `metrics/crop_containment_by_ld_quantile.csv`；
- `metrics/crop_containment_gate.json`；
- 六张要求图，其中 LD–margin 使用 aggregate hexbin，示意图不使用患者影像；
- `reports/crop_containment_report.md`。

## 8. Conditional Stage B

只有 Stage A 为 `GO` 或 `GO_WITH_CAVEAT` 才运行：

- B0：DCE7 → JEPA，无 grounding；
- B1：DCE7 → JEPA + FTV，`lambda_FTV=0.25`；
- B2：DCE7 → JEPA + FTV + LD，两个独立 `Linear(192,1)` head。

首选 2 seeds（2026、4026）×5 folds。只有 input、preprocessing、fold、seed、architecture、source、checkpoint selection 全匹配时才复用 B0/B1，否则重训 matched control。FTV 完全复用 G3 的 fold-train-only log/winsor/robust transform；LD 使用 fold-train-only `log1p`、winsor 与 robust normalization。

`lambda_LD` 只在 seed 2026/fold 0 validation 中按 0.05、0.10、0.25 从小到大选择。最小有效条件为：static LD validation Spearman 相对 B1 至少 +0.03、static FTV 不下降超过 0.03、base validation loss 相对恶化不超过 10%、representation std ≥0.05。不得读取 test 或 pCR。没有候选满足时停止完整 pilot，报告 `LAMBDA_PILOT_FAILURE`，不临时加入 optimization trick。

训练只监督 static FTV/LD，不加入 ΔFTV/ΔLD loss。监控 validation state loss、full base loss、FTV/LD loss、representation std、gradient norm；禁止 PCGrad、warm-up、two-stage 和 gradient normalization。

## 9. Conditional Stage B 评估与决策

冻结 encoder/state 后统一 Ridge：static `r_t→FTV/LD`，longitudinal `Δr→ΔFTV/ΔLD`。主比较 B2−B1，同时保留 B0 reference。重点为 T0/T1/T2 与 T0→T1、T1→T2；T3 与 T2→T3 必须报告但标 floor-risk。SPH/ΔSPH 只作 zero-shot secondary transfer，不训练 SPH。

pCR 仅作 image-only frozen secondary readout：T0 用 `r0`；T0–T1 用 `[r0,r1,r1-r0]`；T0–T2 用 `[r0,r1,r2,r1-r0,r2-r1,r2-r0]`。禁止 clinical、treatment、geometry、radiomics、FTV、LD、mask 与 head output；pCR 不参与 checkpoint/lambda/target 选择。

预注册 effect 口径：

- `DUAL_GROUNDING_GO`：static LD 和 early/mid ΔLD mean Spearman gain 均至少 +0.03，10 个 seed×fold 中至少 7 个方向为正；FTV 与 ΔFTV mean loss 均不超过 0.03；base loss 相对恶化不超过 10%；所有 representation std ≥0.05。
- `STATIC_ONLY_LD`：static LD 达标，但 observed early/mid ΔLD 未达标，且无 harmful 条件。
- `REDUNDANT_OR_HARMFUL`：LD 增益不足，或 FTV/ΔFTV 任一 mean loss >0.03，或 base/representation safety 失败。

pCR 的一致方向变化只作附加证据，不改变上述 decision。

## 10. 隐私、验证与 Git

公开提交只包含代码、配置、aggregate metrics、privacy-safe figures、中文报告和 manifest。禁止提交 patient ID、原始 MRI、原始 Excel、绝对本机路径、大 checkpoint 或 patient-level feature/prediction。

结束前必须扫描公开目录中的 patient-ID pattern 与绝对路径，检查文件大小和 ignore 状态，运行 synthetic crop unit smoke、真实 cache reconstruction smoke、aggregate consistency 与 artifact hash verification。只 `git add` 本实验目录，commit message 固定为 `Add FTV LD dual-grounding pilot`，然后 push `feature/ftv-ld-dual-grounding-pilot`。

## 11. Independent code-review addendum（不改变 gate）

初次全量 Stage A 尚在运行、聚合 gate 结果尚未生成时，独立 code review 指出：`ratio≥0.99` 可能容许极少量、与主体分离的 distal support 落在 crop 外，而该 support 仍可能决定最长空间 extent。为避免高估 LD observability，最终 reproducibility run 增加下列保守敏感性，但不改第 6 节的任何阈值、判定项或方向：

- `exact_full_support_containment`：cached/full support voxel 数完全相等；
- cached/full whole-union 与 largest-component approximate extent retention；
- full bbox 完整落入 recovered crop 的比例；
- origin unique / ambiguous / unresolved 分层和 unique-only LD–margin Spearman；
- DCE 与 FTV mask 的 index-order shape、spacing、slice-first handling 显式一致性断言；
- 公开 LD 分布统计和 hexbin 的 `n<5` small-cell suppression。

这些敏感性只允许强化 NO-GO 或给 GO 增加 caveat，绝不允许把失败 gate 改为通过。物理几何统一表述为 matched-spacing index-space proxy，不宣称 affine/world-space registration。
