# CoRe-WM Shortcut Audit：仓库与现有流程检查报告

检查日期：2026-08-06  
目标分支：`feature/ispy-clean-corejepa`  
目标提交：`c413ec86af04795434bdc19e65bbb006c966f379`  
仓库工作树：`/data/mi2-interns/bowen/Medical_World_Model`

> **后续状态更新（2026-08-06）**：项目方确认原五折资产无法提供，并明确授权参考
> repo 当前模型设计自行重训练。原始资产缺失的事实和本报告检查结论保持不变；后续
> 实验身份改为五折 **audit retraining**，协议见
> `shortcut_audit/report/retraining_protocol.md`，不会宣称复现原 checkpoint。

## 1. 检查结论

仓库结构、数据入口、模型张量流、训练目标、readout 和预期产物已经完成检查；模型代码及共享原始数据可以运行，单患者真实数据 cache smoke 和完整 paper 维度 GPU forward 均通过。

但是，当前不能进入五折 native reproduction，更不能继续作结论性的 shortcut audit。存在三个独立的硬阻塞：

1. clean 分支只实现一次 patient-level 70/15/15 划分，没有五折训练或 OOF 评估入口；
2. clean 专用正式 checkpoint、frozen states、FLR、prediction、metric、threshold 均不存在；
3. 当前 primary FLR 的输入是 `geometry + clinical/treatment condition` 生成的 future response state，完全不读取 MRI latent。这与任务所假设的“以 MRI trajectory representation 做 primary pCR readout”不同。

根据任务约束“native 无法复现时先停止后续结论性分析”，本阶段没有用随机权重、legacy checkpoint 或重新训练结果冒充 native，也没有生成任何虚构指标。

## 2. 分支确认与本地修改保护

最初定位到的目标远程仓库为：

```text
/data/mi2-interns/bowen/Cancer_World_Model/data_source
origin = https://github.com/MI2-Lab/Medical_World_Model.git
```

该现有工作树位于 `feature/ispy-clean-data-processing`，并有 373 项未提交删除记录。为避免切换分支时覆盖或丢失这些修改，采用独立 Git worktree：

```bash
git fetch origin feature/ispy-clean-corejepa
git worktree add -b feature/ispy-clean-corejepa \
  /data/mi2-interns/bowen/Medical_World_Model FETCH_HEAD
```

新工作树确认结果：

```text
branch: feature/ispy-clean-corejepa
commit: c413ec86af04795434bdc19e65bbb006c966f379
commit message: Add clean end-to-end CoRe-JEPA pipeline
```

原工作树及其 373 项本地修改没有被改写、删除或暂存。

## 3. 关键仓库结构

clean CoRe-JEPA 是 `ispy_jepa_tmi_clean/` 下的独立实现：

| 功能 | 关键路径 |
|---|---|
| paper 配置 | `ispy_jepa_tmi_clean/configs/paper_v1.yaml` |
| 数据记录与单次 split | `ispy_jepa_tmi_clean/corejepa/data/records.py` |
| condition 编码 | `ispy_jepa_tmi_clean/corejepa/data/condition.py` |
| DCE8、ROI 与 geometry | `ispy_jepa_tmi_clean/corejepa/data/imaging.py` |
| PyTorch dataset | `ispy_jepa_tmi_clean/corejepa/data/dataset.py` |
| response target | `ispy_jepa_tmi_clean/corejepa/data/response_targets.py` |
| online/EMA encoder | `ispy_jepa_tmi_clean/corejepa/models/encoder.py`、`models/corejepa.py` |
| conditioned transition | `ispy_jepa_tmi_clean/corejepa/models/transition.py` |
| Factorized Response State | `ispy_jepa_tmi_clean/corejepa/models/response_state.py` |
| JEPA 与辅助 loss | `ispy_jepa_tmi_clean/corejepa/training/losses.py` |
| 训练、EMA、checkpoint、state export | `ispy_jepa_tmi_clean/corejepa/training/runner.py` |
| Frozen Landmark Readout | `ispy_jepa_tmi_clean/corejepa/readout/flr.py` |
| cache/pretrain/readout CLI | `ispy_jepa_tmi_clean/scripts/` |
| 数据预处理 | `ispy_jepa_tmi_clean/data_processing/` |
| 方法与张量文档 | `ispy_jepa_tmi_clean/docs/` |

根目录 `requirements.txt` 只覆盖旧数据预处理依赖，没有列出 clean 模型需要的 PyTorch、SciPy、scikit-learn 和 PyYAML。clean 包的实际依赖入口是 `ispy_jepa_tmi_clean/pyproject.toml`。

## 4. 数据路径、样本与 T0–T3 组织

### 4.1 配置路径

`paper_v1.yaml` 指向：

| 资产 | 配置路径 | 状态 |
|---|---|---|
| I-SPY2 root | `/data/data/Preprocessed/I-SPY2` | 存在 |
| I-SPY2 labels | `/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv` | 存在 |
| I-SPY1 root | `/data/data/Preprocessed/I-SPY1` | 存在 |
| I-SPY1 labels | `/data/data/Preprocessed/I-SPY1/clinical_labels_complete4visits.csv` | 存在 |
| clean DCE8 cache | `/data/data/Preprocessed/I-SPY2/_corejepa_clean_dce8` | **缺失** |
| clean response cache | `/data/data/Preprocessed/I-SPY2/corejepa_response_features.npz` | **缺失** |

clean loader 实际得到：

| Cohort / split | 患者数 | pCR 阳性数 |
|---|---:|---:|
| I-SPY2 primary 全部 | 808 | 275 |
| clean primary train | 565 | 192 |
| clean validation | 121 | 41 |
| clean locked test | 122 | 42 |
| I-SPY1 pretraining-only | 156 | 不加载 pCR |
| clean pCR-free pretraining 合计 | 721 | 不用于 endpoint selection |

I-SPY2 的 808 名患者均有四次访视、manifest 和 pCR；I-SPY1 的 156 名患者仅加入 pCR-free pretraining train，不进入 validation、test 或 FLR。

### 4.2 T0–T3 与 tensor contract

数据严格按 `T0,T1,T2,T3` 排列。每名患者 cache 的主要字段为：

```text
image       [4,8,32,96,96]
geometry    [4,9]
condition   [3,25]
```

DCE8 的第 8 个通道就是最终 ROI mask；dataset 没有另行返回独立 mask。9-D `geometry` 由同一裁剪 ROI 计算，包括 volume fraction、bbox 三轴尺寸、bbox volume/fill 及中心位置。

因此扰动实现必须遵循：

- C1 MRI-only replacement：只替换前 7 个 image channels，保留第 8 个 ROI channel 和 9-D geometry；
- C2 full image-derived replacement：替换全部 8 个 channels 和对应 geometry；
- temporal/donor swap：完整 8-channel visit 与 geometry 必须作为整体交换。

### 4.3 ROI 语义与异常来源

I-SPY2 的 analysis mask 不是手工 dense tumor segmentation；零值表示 FTV inclusion region。ROI fallback 顺序由 `corejepa/data/imaging.py` 控制。当前配置保留 `legacy_empty_ftv_full_field: true`：在 66 个 follow-up visit 的 FTV 与 bbox 同时为空时，会退化为 full-field ROI。这些样本必须在 geometry audit 中单独标记，避免把 fallback 行为误认为患者响应。

## 5. Fold manifest 检查

### 5.1 clean 分支实际协议

clean 实现没有五折。`corejepa/data/records.py:90-104` 使用两次 `train_test_split`，`training/runner.py:58-66` 只生成：

```text
primary_train
pretrain_train
validation
test
```

`paper_v1.yaml` 的 `split_seed=2026` 对应一次 70/15/15 划分。仓库内没有 `KFold`、`StratifiedKFold`、fold CLI、fold checkpoint 命名或 OOF 汇总逻辑。

### 5.2 可访问的五折候选副本

发现一份与 808 名 clean I-SPY2 患者及 label 完全一致的五折 manifest 副本：

```text
/data/data/Preprocessed/I-SPY2/
  _matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/
  matched_patient_cv_splits_seed2026.csv
```

校验信息：

```text
rows: 4040
unique patients: 808
each patient appears once as test: true
SHA256: 143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38
```

| fold | train | validation | test | test pCR+ |
|---:|---:|---:|---:|---:|
| 0 | 525 | 121 | 162 | 55 |
| 1 | 525 | 121 | 162 | 55 |
| 2 | 525 | 121 | 162 | 55 |
| 3 | 526 | 121 | 161 | 55 |
| 4 | 526 | 121 | 161 | 55 |

相邻 `summary.json` 声明它复制自：

```text
/home/lin/Projects/Breast_Cancer/ispy2_jepa_world_model/
  runs/core_jepa_5fold_cv_seed2026_20260717/
  patient_cv_splits_seed2026.csv
```

该原始目录目前不存在，且 clean 代码没有引用此副本。因此它是最强的原五折候选，但在拿到 checkpoint 后仍须用 checkpoint 内 patient order/split metadata 校验 provenance，不能直接宣布为 clean 原生 fold。

另有 legacy seed-3072 manifest：

```text
/data/mi2-interns/bowen/Cancer_World_Model/splits/ispy2_5fold_seed3072.csv
```

它同样覆盖 808 人，但其 held-out fold 只有 157/808 人与 seed-2026 候选一致，不能混用。legacy 计划文档还表明该五折是建议补充，并非当时主模型已完成的五折训练。

## 6. 模型真实张量流

### 6.1 Online image state 与 EMA target

`CoReJEPA.encode_visits()` 将两部分相加：

```text
appearance = VisitProjector(VisitEncoder3D(DCE8))
geometry_state = GeometryProjector(q)
visit_state = appearance + geometry_state      # [B,4,192]
```

因此所谓 image latent 本身已经含有两次 lesion 几何信息：DCE8 第 8 个 ROI channel，以及独立 9-D geometry projector。

EMA target encoder 是 online encoder、projector 和 geometry projector 的 deepcopy，全部 stop-gradient；每个 optimizer step 后以 momentum 0.996 更新。target state 同样等于 EMA appearance 加 EMA geometry projection。

### 6.2 Conditioned image transition

`ImageTransition` 使用三层 causal Transformer、四个 attention heads、512-D FFN 和 FiLM conditioning：

```text
T0           -> T1 latent
T0,T1        -> T2 latent
T0,T1,T2     -> T3 latent
```

condition 的 25 维包括：

- 3 维目标访视 one-hot；
- 4 维 observed-prefix mask；
- 14 维 exact treatment-arm one-hot；
- HR、HER2、MammaPrint、标准化年龄。

Transformer 还有 learned positional embedding。因此 nominal time 同时通过 learned position、目标访视 one-hot 和 observed-prefix bits 暴露。

### 6.3 Factorized Response State

`FutureResponseState` 不读取 MRI latent，而是读取 observed `q_0:q_2` 与 condition，输出：

```text
future_state       [B,3,64]
decoded_geometry   [B,3,9]
latent_correction  [B,3,192]
gate logits/prob   [B,3,6]
```

六个 expert 的 gate 去掉前 7 维 temporal condition，但仍读取 treatment arm、HR、HER2、MammaPrint 和年龄；gate 还直接以六类 treatment family 为训练监督。

最终 JEPA 预测为：

```text
prediction = image_prediction + response.latent_correction
```

### 6.4 JEPA distance 与 Copy-current 复用

训练 prediction loss 对每个 192-D prediction/target 分别做 feature-wise LayerNorm，再计算均方误差：

```python
p = F.layer_norm(prediction, (prediction.size(-1),))
t = F.layer_norm(target, (target.size(-1),))
d = ((p - t) ** 2).mean(dim=-1)
```

raw MSE 只记录，不参与 prediction loss。target 通过 `no_grad()` 和 `.detach()` stop-gradient。

Copy-current 主比较应使用：

```text
learned = output.prediction
copy    = output.visit_state[:, :-1]
target  = output.target
```

不能把不含 response correction 的 `output.image_prediction` 当作 learned prediction。建议另报 EMA-current copy 敏感性分析，以分离 online/EMA 表征滞后，但不能替代任务指定主结果。

## 7. Frozen Landmark Readout 的实现事实

这是与任务背景最重要的差异。

`CoReJEPA.forecast_response(geometry, condition)` 只调用 response transition；`training/runner.py` 导出的 `future_response_state` 也只来自 geometry 与 condition。`readout/flr.py` 只加载该 `future_response_state`，完全不加载 `image_prediction`、`visit_state` 或 MRI latent。

primary FLR 的实际路径是：

```text
observed q_0:t + nominal time + clinical + treatment
    -> FutureResponseState
    -> future_response_state
    -> landmark_features
    -> shared class-balanced logistic regression
    -> pCR probability
```

它不是：

```text
MRI latent trajectory -> frozen readout -> pCR
```

直接后果：

- C1 只替换 MRI、保留 geometry 时，primary FLR 的 state 和概率按确定性架构应严格不变；
- temporal/donor swap 中，MRI 部分不会改变 primary FLR，只有被一同交换的 geometry 会改变结果；
- 这类零变化不能被表述为经验上证明“MRI 没有信息”，因为当前 primary readout 根本没有接收 MRI latent；
- copy-current latent audit 与 primary pCR audit 不能直接互相推断，两条路径只在 JEPA 训练时通过 latent correction 间接耦合。

FLR 的其他实现细节：

- 一个 shared logistic regression 同时训练三个 landmark；
- class-balanced、median imputation、standardization、liblinear；
- validation 只选择 penalty 与 C；
- 分类阈值固定为 0.5，不是 validation-selected threshold；
- `flr_scores.csv` 没有 fold、threshold、checkpoint、audit condition 等任务所需字段。

## 8. Checkpoint、readout、prediction 与论文结果

### 8.1 远端与完整历史复核

进一步对 origin 的全部公开 refs 与提交历史做了只读复核。远端只有三个 branch，
没有 tag：

```text
feature/ispy-clean-corejepa        c413ec86af04795434bdc19e65bbb006c966f379
feature/ispy-clean-data-processing 9097a80ba2bcd1aa6c11dc22bc7f3948b1a74402
main                               0f98337c8080ba845a384b4b9eec734fdca0384f
```

远端没有 release；三个分支的历史树、commit message、公开 issue/PR 以及相关关键词
搜索均未发现五折入口、fold-specific clean checkpoint/readout/prediction，或
`core_jepa_5fold_cv_seed2026_20260717` 的任何引用。模型代码只在当前
`c413ec8` 首次加入；其 `pretrain.py` 只调用一次 train，配置只有一个输出目录，
records/runner 也只构造一组 70/15/15 split。

因此上述带日期的五折运行名更像仓库外本地 run 或未提交 wrapper。即使以后找到同名
目录，也必须检查 payload 的 clean schema 与逐折 patient IDs，不能仅凭目录名认定
provenance。

### 8.2 当前文件系统中的预期产物

clean 预期输出目录为：

```text
ispy_jepa_tmi_clean/runs/corejepa_clean/
```

预期文件：

```text
config.yaml
history.csv
best_corejepa.pt
last_corejepa.pt
frozen_states.npz
splits.json
flr.pkl
flr_metrics.csv
flr_scores.csv
flr_summary.json
```

实际为 0/10；该目录本身不存在。对 `/data` 和 `/home/bowen` 进行精确文件名搜索也没有找到任何 clean `best_corejepa.pt`、`frozen_states.npz`、`flr.pkl` 或 `flr_*` 产物。

Cancer 开发树中存在大量 legacy checkpoint，包括 E30a `best_loss.ckpt`，但：

- 没有 fold-specific checkpoint；
- E30a 是 seed-3072 单次 split；
- 参数名和 payload schema 与 clean 模型不同；
- clean README 明确说明 legacy development checkpoint 不能按名称加载。

因此不能用 E30a 或已有 legacy shortcut audit 冒充 clean native reproduction。

clean 仓库还没有给出 paper T0、T0–T1、T0–T2 的数值结果或 prediction CSV，无法执行“与论文结果在合理误差内比较”。

## 9. 原始命令与运行环境

官方命令必须从 `ispy_jepa_tmi_clean/` 执行，因为 metadata 路径是相对路径：

```bash
cd /data/mi2-interns/bowen/Medical_World_Model/ispy_jepa_tmi_clean
python scripts/build_tensor_cache.py --config configs/paper_v1.yaml
python scripts/build_response_cache.py --config configs/paper_v1.yaml
python scripts/pretrain.py --config configs/paper_v1.yaml
python scripts/fit_readout.py --config configs/paper_v1.yaml
```

当前 base Python 缺少 PyTorch 和 pydicom；可用环境是：

```text
/home/bowen/.conda/envs/bowen
Python 3.11.14
PyTorch 2.9.1+cu130
CUDA available: true
GPU count: 3
```

本机等价命令应加 `conda run -n bowen`。该环境缺可选测试依赖 pytest，因此 `python -m pytest` 当前不能直接运行；模型和真实数据 smoke 不受此影响。

## 10. 已完成 smoke checks

### 10.1 Synthetic GPU model smoke

命令：

```bash
conda run -n bowen python scripts/smoke_model.py --gpus 0 --batch-size 4
```

结果：

```text
prediction=(4,3,32)
response_state=(4,3,16)
forward/loss/backward: passed
```

### 10.2 真实患者 tensor/response cache smoke

患者 `ACRIN-6698-102212`：

```text
image:          [4,8,32,96,96] float32, finite
geometry:       [4,9] float32, finite
response cache: [1,4,106] float32
```

smoke 产物保存在 `shortcut_audit/logs/cache_smoke/`，没有写入或覆盖共享正式 cache。

### 10.3 完整 paper 维度真实数据 forward

在 GPU 0 上成功得到：

```text
image:                 [1,4,8,32,96,96]
condition:             [1,3,25]
prediction:            [1,3,192]
EMA target:            [1,3,192]
future response state: [1,3,64]
all finite: true
```

这说明当前阻塞来自正式产物与协议缺失，而不是共享原始数据或模型 forward 损坏。

## 11. 各 audit 计划复用或包装的函数

| Audit | 复用入口 | 必要 wrapper / 注意事项 |
|---|---|---|
| Native | `build_datasets`、`export_frozen_states`、`landmark_features`、已冻结 FLR | 只加载已有 fold checkpoint/readout；绝不调用 fit 覆盖原 readout |
| Copy-current | `CoReJEPA.forward()` | 精确复刻 LayerNorm-MSE；主 learned 使用 combined `prediction`；按患者聚合 transition |
| Repeated-T0 C1 | `LongitudinalDCEDataset`、`forecast_response` | 只复制 image channels 0–6；primary FLR 预期结构性不变 |
| Repeated-T0 C2 | 同上 | 复制全部 8 channels 与 geometry；保留 nominal condition |
| Temporal order | `encode_visits`、`encode_targets`、`image_transition`、`response_transition` | perturb prediction 与 **native target** 比较，不能让 swapped target 随输入一起换位 |
| Matched follow-up swap | `treatment_family`、cache tensor、`landmark_features` | donor 只用 baseline observable 匹配；保留 recipient condition；保存完整 mapping |
| Clinical baseline | `PatientRecord`、`ConditionEncoder` 中临床字段 | 每 fold 只在 train/validation 选择超参数，test 仅最终评估 |
| Geometry baseline | cache `geometry` | 只使用决策点可见 q 与变化；记录 full-field fallback |
| Static T0 imaging | `encode_visits` 或冻结 T0 representation | 需要明确的 MRI-latent readout 协议；不能冒充当前 geometry-only primary FLR |

所有 paired perturbation 必须使用 `model.eval()`，否则 dropout 会制造伪概率差异。

## 12. 实现与任务/论文描述之间的差异

1. **五折 vs 单次 split**：任务要求五折 OOF；clean 代码只实现一次 70/15/15。
2. **primary readout 输入**：任务假设 MRI trajectory representation；clean primary FLR 只读取 geometry/condition response state。
3. **阈值**：任务要求 validation-selected threshold；clean 固定 0.5。
4. **checkpoint**：任务要求复用已训练 fold checkpoint；clean 仓库和共享路径都没有对应产物。
5. **legacy 兼容性**：clean README 明确不按名称加载 development checkpoint。
6. **condition 统计**：`ConditionEncoder` 的 arm vocabulary 与年龄均值/标准差由全部 964 条 mixed records 构建，不是每 fold train-only；checkpoint 虽保存 metadata，却没有恢复该 encoder 的现成接口。
7. **target latent 含 geometry**：EMA target 并非纯 image state，而是 EMA appearance 与 EMA geometry projection 之和。
8. **时间信号冗余**：learned position、target one-hot 和 prefix bits 同时暴露 nominal time。
9. **mask 与 geometry 双重输入**：ROI 同时作为 DCE8 第 8 channel 和独立 9-D projector 输入。
10. **输出 schema**：原 `flr_scores.csv` 不满足审计所需的 fold、threshold、checkpoint 和 condition 字段。

## 13. Native reproduction 阻塞与所需最小资产

继续前至少需要项目方提供或确认：

1. 原 `core_jepa_5fold_cv_seed2026_20260717` 的五套 fold-specific clean checkpoint；
2. 每折对应的 train/validation/test manifest、patient order 与 response-target transform；
3. 每折 frozen FLR/readout、validation-selected threshold（若确实存在）和 native prediction；
4. 原论文/提交版 T0、T0–T1、T0–T2 的参考数值、配置和命令；
5. 明确 primary endpoint 应审计当前 geometry-only FLR，还是尚未提交的 MRI-latent FLR。

若项目方确认允许重新训练，仍需另行明确：

- 是否采用上述 seed-2026 五折候选 manifest；
- 是否每折重新训练 representation，还是共享一个 pCR-free representation checkpoint；
- readout 的 validation threshold 选择规则；
- 这会产生新的 audit reproduction，而不是“复用原始已训练 checkpoint”。

在这些问题解决前，不应运行或解释 B–F 的正式数值结果。

## 14. 可重跑前置条件检查

已新增只读检查入口：

```bash
conda run -n bowen python shortcut_audit/scripts/check_prerequisites.py
```

机器可读结果保存在：

```text
shortcut_audit/metrics/prerequisite_check.json
```

当前状态为 `blocked`，列出的失败项与本报告一致。

## 15. 已实现但尚未运行正式数据的审计保护层

为减少资产恢复后的实现歧义，已在 `shortcut_audit/auditlib/` 中完成以下独立
wrapper；它们没有改动 clean 训练代码，也没有生成正式 B–F 数值：

- `folds.py`：五折 patient-level、OOF 唯一性、label 与 patient order 校验；
- `provenance.py` / `runtime.py`：clean checkpoint schema、逐折 split、condition
  metadata、严格模型恢复与冻结；
- `perturbations.py`：C1/C2、T1/T2 swap、donor T1/T2 replacement；latent audit
  明确用未扰动序列生成 EMA target；
- `matching.py`：held-out fold 内、禁止 self donor、outcome-blind 的 subtype / treatment /
  visit hard matching 与 baseline volume soft distance；
- `metrics.py`：原 JEPA feature-wise LayerNorm-MSE、copy gain、fold/pooled 指标、
  patient 等权聚合与 paired patient-block bootstrap；
- `contracts.py`：prediction CSV schema、threshold/label/checkpoint/跨 condition 对齐及
  原子写入。

统一命令：

```bash
conda run -n bowen python -m unittest discover -s shortcut_audit/tests -v
```

当前共 53 项测试通过，包括真实 clean model import、checkpoint/split 负向校验、
NumPy/Torch 扰动 clone 安全性、native target 固定、donor fold 隔离、JEPA 距离数值
定义、patient-block bootstrap 和 CSV 契约。另有一项结构性测试确认：C1 改变 MRI
但保留 geometry/condition 时，当前 `forecast_response` state 必须完全不变。这是代码
路径事实，不是五折经验结果，也不解除 native reproduction 门槛。

资产契约及核验顺序另见 `report/reproduction_asset_contract.md`。
