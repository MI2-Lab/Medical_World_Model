# Local–Global Response State Pilot

## 1. 目标与单一干预

本实验检验：在 response-observable large-FOV 的 C1B-H 输入不变时，显式保留局部 treatment-response information，能否把 frozen pooling audit 的 local advantage 转化为端到端学习的 response state 及 observed longitudinal dynamics。

唯一改变的模型区段是：

```text
encoder.features[3] final spatial map -> spatial aggregation -> 192-D response state
```

输入、3-D encoder、JEPA projector/transition、target encoder/EMA、SIGReg、optimizer、FTV target、fold、seed、logical batch 与 checkpoint selection 均冻结。唯一 grounding target 是 FTV_t；`lambda_FTV=0.25`。

本实验不是 crop、encoder、attention、optimization stabilization、pCR 或 FTV+LD 实验。禁止 LD、SPH、BPE、Delta-FTV supervision、pCR、clinical/treatment supervision 与 learned spatial attention。

## 2. 冻结的上游证据和数据

- Stage A sentinel：`c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json`，SHA-256 `0b2c9e...afbdb`。
- Stage B private data contract：SHA-256 `dd22f1...ab27`；原位只读复用，不复制 947 个 C1B cache。
- technical eligibility：947 patients。
- formal primary population：375 patients / 1500 visits；1486 visits grounding-observable。
- C1B-H DCE7：`float32 [4,7,112,176,160]`，spacing XYZ = `[0.9,0.9,2.0]` mm。
- folds 与 fold-external train-only population 原样复用；不得 refill 或移动 patient。
- lock 的 upstream inventory 覆盖所有实际自动执行的 package `__init__.py`，并完整覆盖复用的 Stage-B `contracts/data/gate/inputs/targets/training/upstream`、G3 `config/data/model/training/targets` 及 audited pooling 实现；任一 import-closure hash 漂移均禁止执行。

完整冻结项、路径和 hash 见 [pilot.json](configs/pilot.json)。正式训练前以独立 lock 文件绑定 plan、config、新代码与上游代码的 SHA-256。

## 3. Spatial response-state architectures

运行时从实际 `encoder.features[3]` output 取得 `F_t`；不硬编码 `14x22x20`，但必须验证 channel=128 及 frozen convolution geometry。

### GAP

`z_G = mean(F_t, D/H/W)`，随后使用原 `Linear(128,192)+LayerNorm(192)`。

### LOCAL

`z_L` 使用上一轮正式 audit 的固定中心 `64x64x64 mm` physical cube。feature cell weight 是其 stride-8 sampling cell 与该 cube 的 separable fractional overlap，随后严格 normalized weighted mean；无 epsilon、无 empty fallback。T0-T3 共用同一坐标约定；不读取 lesion/valid-source mask、FTV 或 outcome。

### LOCAL_GLOBAL

先在 raw pooled space 拼接 `[z_L;z_G]`（256-D），再使用唯一的 `Linear(256,192)+LayerNorm(192)`。不得误用上一轮 audit 中先分别投影再拼接成 384-D 的 probe state。

同一 effective seed 下，先产生标准 baseline `W,b`。GAP 与 LOCAL 使用完全相同的 `W,b`/LayerNorm；LG 使用 `[0.5W,0.5W]`、bias=`b` 与相同 LayerNorm。因此当 `z_L=z_G` 时，初始 LG response 与 baseline 精确相同。在线和 EMA target 路径必须对称采用相同 architecture。

## 4. Formal matrix

| Arm | Pooling | Grounding |
|---|---|---|
| GAP0 | GAP | none |
| GAP3 | GAP | Direct FTV |
| LOCAL0 | fixed 64-mm LOCAL | none |
| LOCAL3 | fixed 64-mm LOCAL | Direct FTV |
| LG0 | raw LOCAL+GAP concat | none |
| LG3 | raw LOCAL+GAP concat | Direct FTV |

矩阵为 2 seeds（2026、3026）x 5 folds x 6 arms = 60 formal cells。所有 arm 只使用 C1B-H；legacy L1/L3 只作 historical reference，不重新训练。

## 5. Training 和 selection

- physical batch 4，accumulation 8，logical batch 32。
- SIGReg 在完整 B32 上作一次 nonlinear reduction，并通过已验证 surrogate 给出 exact accumulated gradient。
- 每 logical batch 仅一次 gradient clip、AdamW step、EMA update。
- epochs 12，patience 4，LR `5e-5`，weight decay `1e-4`，EMA `0.996`，clip `5.0`。
- grounded FTV transform 仅在 outer-train observable visits 拟合：log、1/99 winsorization、median/IQR。
- `grounding_observable_mask` 只进入 loss；non-observable patient/visit 仍参加 JEPA。
- no-grounding checkpoint：non-collapsed finite epochs 中 minimum validation state loss。
- grounded checkpoint：先要求 validation state loss 不超过 paired no-ground selection 的 1.05 倍，再在合格 epoch 中取 minimum validation FTV loss。test FTV、Delta-FTV、pCR 不得参与 selection。
- 4/8 OOM 时立即停止；本预注册不授权 2/16 或任何 fallback。若需改变 batch contract，必须先创建新的代码版本与新预注册，并使用新的空 root；禁止 partial fallback/resume。

## 6. Frozen probes

selected online pre-projector `r_t [N,4,192]` 冻结后，严格 outer-fold-isolated Ridge：X scaler outer-train only；alpha grid `1e-4...1e3`，用 validation analysis-space MSE 选择，tie 取最小 alpha；test 只 predict 一次，不 refit。

Static FTV 报告 T0-T3 与 macro：Spearman、Pearson、natural/transformed R2、RMSE、MAE、B0 RMSE gain、prediction/target variance ratio、descriptive calibration slope。

Observed Delta-FTV 只在冻结表征后定义 `Delta-r_t=r_(t+1)-r_t`，预测 literal natural `FTV_(t+1)-FTV_t`，报告三 transition 与 macro 的 Spearman、Pearson、natural R2、RMSE、variance ratio。训练阶段禁止 Delta-FTV supervision。

Natural metrics 先合并五个 outer-test folds 再计算 endpoint；macro 是 endpoint metric 的无权均值。各 fold transform 不可直接合并，因此 transformed metrics 只作 fold summary。

## 7. 预注册 Gates 与选择规则

精确机器可读定义在 `pilot.json`。为消除“系统性恶化/明显改善”的歧义，正式结果前冻结以下 operationalization：

- natural R2 systematic worsening = 两个 seed 的 effect 均严格小于 0；
- meaningful Delta-FTV Spearman gain = 至少 `+0.02`（沿用 Gate B 的 effect-size 尺度）；
- threshold 仅是 pilot decision rule，不是统计显著性；fold 是 paired sensitivity，不是独立 replicate。

Gate A：每个 seed `LOCAL0-GAP0` static macro Spearman 至少 +0.10，Delta macro Spearman 严格改善，且 static natural R2 不在两个 seed 均恶化。

Gate B：每个 seed `LG0-LOCAL0` static macro Spearman 非负，至少一个 seed 达 +0.02，且 natural R2 不系统性恶化。

Gate C：最终 candidate 的 grounded-base static macro Spearman 在两个 seed 均严格改善，至少一个 seed 的 Delta macro Spearman 达 +0.02，且 natural R2 不系统性恶化。

Gate D：最终 grounded candidate 至少 9/10 paired folds 的 selected validation state-loss degradation 不超过 5%。

Architecture rule：A 通过而 B 不通过选 LOCAL；A、B 均通过选 LOCAL_GLOBAL；A 不通过则不得进入 FTV+LD，并分类为 frozen advantage did not transfer。若 architecture 改善而 final grounded candidate 不安全，分类 D 优先。

## 8. 必须产物

Tables：architecture contract；static FTV；observed Delta-FTV；paired architecture effects；grounding effects；optimization safety；prediction variance/calibration。

Figures：architecture schematic；static Spearman；static natural R2；Delta Spearman；Delta natural R2；variance ratio；calibration slope；paired fold effects；safety heatmap；representative training curves。

最终生成中文 `reports/final_report.md`，逐项回答 brief 的 13 个问题，并只使用 A/B/C/D 中一个主要 scientific classification。

即使 A 或 B 通过，本 pilot 也不直接授权 FTV+LD：先完成本报告，再进行 selected architecture 的更大 multi-seed confirmation；确认稳定后才设计 FTV+LD Factorized Grounding Pilot。
