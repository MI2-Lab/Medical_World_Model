# Observed-State Radiomics Decodability Audit 实验计划

日期：2026-08-07

## 1. 科学问题与诊断目标

上一轮 M1 改善了 normalized copy-current transition gain，但 M2 的 predicted-delta radiomics head 近常数、pCR 也未改善。本实验绕过 transition，不训练任何 world model，直接检验真实 observed MRI encoder representation 中是否存在可线性解码的当前 measurement 与纵向变化信息。

需要回答：

1. global observed latent 是否包含当前 FTV、LD、sphericity 信息；
2. 两个真实 observed latent 的差是否包含相邻 measurement change；
3. 信息是否存在于 global pooling 前的 ROI/local feature，却在 global compression 中丢失；
4. M1/M2 是否改变了 encoder representation，而不仅是 transition；
5. 瓶颈主要位于 encoder、pooling/projector、transition、target，还是多者共同作用。

BPE 因当前 lesion-centered crop 未必覆盖足够背景乳腺组织，只作为探索性 target，失败不作为 encoder 失败的主证据。

## 2. 冻结资产

使用 `m0_final`、`m1_final`、`m2_final` 的 fold 0–4 `best.pt`，不重新训练或微调 encoder、projector、transition、radiomics head。每名 test 患者只使用其唯一 test fold 的 checkpoint。详细 SHA、epoch、fold、cache 和数据版本见 `reports/asset_inspection.md`。

所有 checkpoint 是 8 通道 DCE7+ROI-mask 模型，没有独立 geometry/clinical/treatment/radiomics 输入。本报告必须使用“ROI辅助 observed image representation”，不能称为严格纯影像。

## 3. 为什么绕过 transition

M2 原生 head 的输入是 transition 预测的 `predicted_delta`。该 head 失败可能有两种完全不同的来源：

- encoder 根本没有保留 treatment-response measurement 信息；
- encoder 已保留信息，但 transition 无法把 observed trajectory 映射到正确的 future delta。

以真实 T0–T3 observed states 构造 static 与 adjacent-delta probe，可将 image representation 与 transition prediction 解耦。这里的 T3 只作为真实 observed MRI/相邻 T2→T3 audit 数据，不用于重新训练任何临床模型。

## 4. Observed representation

实际计算图为 `[8,32,96,96] → [128,4,12,12] → GAP 128 → Linear+LN 192 → projector 192`。主分析使用直接被训练、且被 transition 读取的 online 分支；EMA target 分支作为上一轮 current-state/readout 坐标的敏感性分析。

核心 representation：

- R1 `online_projected`：192-D projected global latent；
- R2 `online_preprojector`：192-D projector 输入；
- R2b `online_global_pool`：128-D GAP 输出；
- R4 `online_roi_mean`：128-D ROI occupancy-weighted local feature；
- 相同四项 EMA target sensitivity；
- B2 `mask_geometry`：ROI volume、normalized z/y/x bbox extent、bbox diagonal、normalized z/y/x centroid，共 9-D；
- B3 `raw_roi_intensity`：DCE7 各通道 ROI mean/std，共 14-D。

不长期保存完整 `[128,4,12,12]` tensor。ROI mask 用 adaptive-average occupancy resize 到 `[4,12,12]` 后加权 mean pooling。空 mask cell 标为无效，并从 ROI probe 的对应 static visit 或 change pair 中剔除；global probe仍保留这些患者。

## 5. Target 定义

### 5.1 Static measurement

从上一轮已核验的四访绝对值重建 `FTV_t`、`LD_t`、`sphericity_t`、`BPE_t`。每折、每 timepoint、每 feature 只用 outer-train 配对患者拟合：

- FTV/LD：使用该折已有 epsilon 做 `log(value+epsilon)`；
- sphericity/BPE：identity；
- 1%/99% winsorization；
- median/IQR robust standardization。

生成独立 `configs/static_target_transform_fold_<k>.json`，记录 train patient hash、raw-target hash、timepoint、epsilon、边界、center、scale 和样本数。Validation/test 只应用保存参数。

### 5.2 Adjacent change

为与上一轮 M2 supervision 精确对应，直接复用每折已有的 `radiomics_transform_fold_<k>.json`：

- ΔFTV、ΔLD：`log(end+epsilon)-log(start+epsilon)`；
- Δsphericity、ΔBPE：`end-start`；
- train-only winsorization 与 median/IQR standardization。

主分析三个相邻 transition 为 T0→T1、T1→T2、T2→T3，不创建虚假的缺失 target，也不使用 pCR。

## 6. Probe 矩阵

### 6.1 Static

对每个模型、outer fold、timepoint、target 分别拟合：

- projected global → current measurement；
- pre-projector global → current measurement；
- GAP global → current measurement；
- ROI mean → current measurement；
- EMA 对应项作为敏感性。

### 6.2 Longitudinal change

对每个 representation family 和相邻 transition 构造：

- D1 current-only：`x_t`；
- D2 observed pair：`[x_t,x_(t+1)]`；
- D3 observed difference：`x_(t+1)-x_t`；
- D4 combined：`[x_t,x_(t+1),x_(t+1)-x_t]`；
- D5 ROI difference：ROI family 的 D3；
- D6 ROI combined：ROI family 的 D4。

D4 与 D2、D6 与 ROI pair 在线性代数上严格冗余；仍按目标要求输出，但不把 Ridge 正则化参数化造成的差异解释为新增信息。核心 transition 诊断以 D3 global delta 对 D5 ROI delta为主。

## 7. Probe、split 与选择规则

主 probe 为线性 Ridge；不使用复杂 MLP 强行提高结果。Alpha 网格固定为 `{1e-4,1e-3,1e-2,1e-1,1,10,100,1000}`。

对每个 outer fold：

1. outer-train patient rows 拟合 StandardScaler 和 Ridge；
2. outer-validation 只选择 alpha，目标是 standardized MSE 最小，1e-12 内并列取更小 alpha；
3. 锁定后才预测 outer-test；
4. 同一患者的所有 visit/transition始终留在同一 split；
5. 不跨 fold 混合 test feature训练 probe，不以 test 选择 alpha、transform、PCA 或阈值。

Probe 按 timepoint/transition 分开拟合，避免治疗阶段均值差异伪装成 image decodability。M0/M1/M2 对每个 cell 使用完全相同 patient mask 和相同 target transform。

## 8. Baseline 与控制

- B0 train-mean：每折、timepoint/transition、feature 只用 train target 均值，直接用于 validation/test；change 另记录 zero-change comparator。
- B1 current-radiomics：`r_t → Δr`，使用相同 train/validation/test 和 Ridge 选择规则；它是 table-dependent response baseline，不支持 image-only 主张。
- B2 mask geometry：static 使用当前 9-D geometry；change 报 current、pair、delta、combined。它用于判断 image/local feature是否仅达到 ROI mask 已显式提供的信息。
- B3 raw ROI intensity：14-D DCE7 ROI mean/std，作为简单影像 baseline。

所有 baseline 与模型 probe 在同一有效 test row上比较；ROI empty cell 不以零向量补齐。

## 9. 指标与 near-constant 防护

每个 fold、pooled OOF、timepoint/transition分别报告：MAE、RMSE、R²、Spearman、Pearson、prediction/target variance ratio。Change 另报方向准确率和相对 train-mean RMSE gain。

Near-constant 定义为预测样本方差 `≤ max(1e-10,0.01×target variance)`。Pooled OOF 额外报告按 fold 去中心后的 variance ratio，防止不同 fold 截距人为抬高预测方差。Bootstrap 使用 2,000 次 patient-cluster resampling；pooled static/change 中一个患者的全部行作为同一重采样单元。

主判断标准不是单个 loss，而是联合要求：

- Spearman 方向稳定且 bootstrap CI 不以零为中心；
- R² 或相对 train-mean RMSE gain为正；
- 预测不是 near-constant；
- 多个 fold/transition 不是由单一 cell 驱动。

## 10. 输出

- `features/<model>/fold_<k>/observed_features.npz`：冻结 representation；
- `features/feature_manifest.csv`：patient/model/fold/split/timepoint/representation/file/hash/有效性；
- `predictions/`：所有 outer-test prediction-level CSV；
- `metrics/`：fold-level、pooled OOF、bootstrap CI、variance、baseline 与 paired difference；
- `figures/`：global-vs-ROI、model comparison、三个主 target scatter、R²/variance heatmap、representation difference、transition-specific 图；
- `reports/final_report.md`：中文瓶颈诊断。

Patient-level feature/prediction 和含 trial ID 的 target 保持在本地 `.gitignore` 范围，不写回或覆盖上一轮目录。

## 11. 诊断映射

| 观察 | 诊断 |
|---|---|
| Global 与 ROI static/change 都不可解码 | encoder或输入/target spatial mismatch为主；不继续优化 transition |
| ROI 明显优于 global | global pooling、encoder output或projector丢失 spatial response 信息 |
| Observed global/ROI delta 可解码，但上一轮 predicted delta失败 | transition/trajectory learning是主要瓶颈 |
| M2 observed representation稳定优于 M1/M0 | privileged supervision改变了 encoder，只是原 head/transition使用失败 |
| M0/M1/M2近似 | objective主要改变 transition；应把 grounding 更直接施加到 observed local feature |

最终结论允许“多瓶颈共同作用”，但必须由 static、delta、global/ROI、mask baseline、模型差异和 transition-specific 证据共同支持，不能仅凭 pooled 点估计。

## 12. 停止边界

本轮不训练或微调 world model，不扫 `lambda_rad`，不运行 M3，不新增 Transformer，不训练 pCR 主模型。Peritumoral ring、ElasticNet 和 MLP属于可选低优先级；只有核心 linear global/ROI 矩阵完成后且能增加诊断价值时才考虑。Frozen ROI linear head 与 ROI Ridge probe等价，若线性结果已经充分则不另做重复 pilot。
