# Direct Grounded Response State 实验计划

## 0. 计划状态与不可变边界

本计划在正式训练前写定。执行分支为 `feature/ispy-clean-corejepa`，计划创建时 HEAD 为 `629b9cdb6d9a713ca03cc7ff700c8d2fd71dc960`。运行环境固定为 conda `bowen`、Python 3.11.14、PyTorch 2.9.1+cu130、CUDA 13.0；机器有 3 张 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（每张约 96 GiB）。GPU 同时存在其他用户作业，训练不假定独占设备。

本实验只改变三件事：是否把 binary ROI mask 作为图像通道、是否用 mask 做归一化 ROI pooling、是否在真实 observed state 上施加 static FTV grounding。以下项目保持不变：M0-style Next-State/JEPA transition、EMA target、SIGReg、transition step weights、训练 fold、数据预算和 image-only downstream 约束。

明确禁止：Next-Change、direction/magnitude/path loss、relational loss、clinical/treatment/9-D geometry 输入、pCR-supervised encoder、FTV/radiomics inference input，以及把 mask 偷渡回 G3/G4 的 backbone。所有新增代码与产物只写入本目录；原始数据、manifest、旧 checkpoint、`shortcut_audit/`、`radiomics_next_change/`、`observed_state_radiomics_audit/` 均只读。

## 1. 已有证据与本轮动机

已完整核对以下两份报告：

- `radiomics_next_change/reports/final_report.md`，SHA-256 `4fad96d1f6fa3bae3c879ec16d57699423836d81c415273fd86f0eec619f54ef`；
- `observed_state_radiomics_audit/reports/final_report.md`，SHA-256 `c450d4800a6644f0cca684d3f8e79b670990946105c6f18c6f911e472a26f8c6`。

与本轮直接相关的已知事实是：

1. M0 observed representation 可稳定解码 static FTV；pre-projector 通常略优于 projected latent。
2. M0 observed latent difference 对 ΔFTV 有有限但稳定的信号；ΔLD/Δsphericity 明显更弱。
3. M1 Next-Change 没改善 observed representation；M2 对 predicted delta 的 auxiliary head 基本没有改变 M1 representation。
4. Frozen transition-predicted delta 无法稳定恢复 observed ΔFTV signal，因此本轮不 ground predicted delta。
5. ROI pooling 没有普遍优于 GAP。
6. 9-D mask geometry 对 static FTV 与 ΔFTV 极强，旧数据审计中 FTV 与 mask voxel count 的 Spearman 为 0.935；mask shortcut 是本轮的核心混杂。

因此，本轮把监督前移到真实 observed image state：先检验 `MRI_t -> r_t -> FTV_t`，再在训练未见过 ΔFTV loss 的条件下检验 `r_(t+1)-r_t` 是否自然获得 response meaning。

## 2. 数据、DCE 通道与五折

### 2.1 DCE8 与 DCE7

legacy cache 中每个患者的 `x` 为 `[4,8,32,96,96]`，访视顺序为 T0/T1/T2/T3。八个通道依次为：

| 索引 | 定义 |
|---:|---|
| 0 | pre-contrast |
| 1 | early post-contrast |
| 2 | late post-contrast |
| 3 | early − pre |
| 4 | late − pre |
| 5 | peak relative enhancement |
| 6 | washout relative enhancement |
| 7 | binary ROI mask |

DCE7 严格定义为 `x[:, :7]`。G2/G4 的 loader 同一次只读 NPZ 后另取 `x[:, 7:8]` 为命名清楚的 `roi_mask`；mask 与 DCE tensor 永不拼接。需要诚实保留的边界是：现有 `[32,96,96]` cache 是 lesion-centered crop，crop 定位曾使用 ROI，因此“DCE7-only”表示模型时不输入 mask channel，而不是从未使用任何 ROI 先验的全乳 MRI。

### 2.2 Cohort 与 split

- primary cohort：808 名 complete-four-visit I-SPY2，均有 pCR label；
- extra base pretraining：156 名 complete-four-visit I-SPY1，只贡献 base loss；
- measurement matched：375/808 名 I-SPY2，四访 FTV/LD/sphericity/BPE 完整；
- 433 名无 measurement 的 I-SPY2 仍贡献 base loss，不填造 FTV、不删除、不过采样。

五折固定使用：

`${DGRS_DATA_ROOT}/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv`

SHA-256 为 `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。fold 0–2 的 train/val/test 为 525/121/162，fold 3–4 为 526/121/161；每名患者恰好一次进入 test。它是经过完整一致性检查的 seed-2026 candidate copy，但缺少 clean native split 的原始生成 provenance，因此报告不会声称 native numerical reproduction。

## 3. 当前 M0 encoder 的真实计算图

现有可复用 M0 是 `additional_experiments/radiomics_next_change` 的无 clinical/condition/9-D geometry image-only Next-State 模型，不是 clean `CoReJEPA` 中带 geometry/condition correction 的主模型。

单访视实际图如下：

```text
DCE [C,32,96,96]
-> ResidualBlock C->16, stride 1
-> ResidualBlock 16->32, stride 2
-> ResidualBlock 32->64, stride 2
-> ResidualBlock 64->128, stride 2
-> spatial feature map F [128,4,12,12]
-> GAP 或 normalized occupancy-weighted ROI mean [128]
-> Linear(128,192) + LayerNorm
-> pre-projector observed response state r [192]
-> Linear(192,384) + LayerNorm + GELU + Linear(384,192)
-> projected JEPA state z [192]
```

本轮 primary representation 固定为 online pre-projector `r`。它既与前轮表现较好的层对齐，也能让 FTV gradient 直接回传至 3-D encoder。projected `z` 继续进入完全相同的 M0 causal transition；不会删除 projector、把 SIGReg 改到别层或改 transition objective。

## 4. GAP 与 normalized ROI pooling

G1/G3 使用：

```text
f_GAP = mean(F, spatial axes)
```

G2/G4 先用 `adaptive_avg_pool3d` 把 binary mask resize 为 `[1,4,12,12]` occupancy，再使用：

```text
f_ROI = sum(F * occupancy) / max(sum(occupancy), epsilon)
```

非空 mask 的 denominator 只用于归一化除法，不返回、拼接或输入任何 learned module。空 mask 回退 GAP，并只把 `roi_valid=false` 写入诊断。禁止 sum pooling、mask voxel count、bbox、centroid、fill ratio、9-D mask geometry、显式 volume 或 valid flag 进入 `r/z/transition/FTV head/pCR readout`。

归一化均值只消除了直接的 sum-volume 幅值路径；mask support 仍决定采样位置，不能宣称 G2/G4 与几何完全独立。

## 5. G0–G4 矩阵

| 模型 | Backbone 输入 | Pooling | Direct grounding | 主要角色 |
|---|---|---|---|---|
| G0 | DCE7 + mask channel | GAP | 无 | 既有 ROI-assisted M0 baseline |
| G1 | DCE7 | GAP | 无 | 无 mask channel baseline |
| G2 | DCE7；mask 与 backbone 分离 | normalized ROI mean | 无 | 定位-only pooling baseline |
| G3 | DCE7 | GAP | static FTV | 严格模型输入 DCE7 grounding |
| G4 | DCE7；mask 与 backbone 分离 | normalized ROI mean | static FTV | primary candidate |

G0 优先复用 `radiomics_next_change/checkpoints/m0_final/fold_<k>/best.pt` 及 OSRA 的 `online_preprojector` features；五折 best epoch 为 3/3/2/2/3。G0 的旧 pCR readout 使用 transition feature，和本轮 contract 不同，故本轮必须从 frozen `r` 重新拟合 pCR readout。

G1/G3、G2/G4 分别使用相同 seed、patient order 与共享参数初始化；FTV head 在隔离 RNG 中初始化，以保证 grounding 是成对模型间唯一优化差异。

## 6. 不变的 M0-style base objective

四访 online `r` 经既有 projector 得 `z_online`，EMA encoder/projector 得 `z_target`。未加 clinical、treatment 或 geometry condition 的原 causal Transformer 用 `z_online[:, :-1]` 预测三个完整 next states。base loss 为：

```text
L_base = weighted normalized next-state MSE + 0.09 * L_SIGReg
```

预测与 EMA target 沿 latent 维分别 LayerNorm 后算 MSE；三个 transition 的 raw step weights 为 `[2.0,1.0,0.5]`，除以其均值；SIGReg 保持 256 projections。EMA momentum 0.996。G1–G4 均不训练 Next-Change，也不改变 transition 结构。

## 7. FTV transform 与 grounding loss

### 7.1 训练 target

每个 outer fold 新建 `configs/ftv_transform_fold_<k>.json`。为支持四访共享的单一 Linear head，训练 transform 将该 fold train 中有 FTV 的患者四访值合并拟合：

1. `epsilon = max(1e-6, 0.5 * minimum positive training FTV)`；
2. `log(FTV + epsilon)`；
3. 1%/99% winsorization；
4. winsor 后 median 与 `max(IQR,1e-6)`；
5. 标准化。

JSON 保存 fold、train patient hash、raw target hash、总 train/paired/valid visit 数和全部参数。validation/test 只 apply。probe 为了与既有 audit 可比，仍使用已经验证的 per-timepoint static transform；训练 head transform 与 probe transform用途分离并显式记录。

### 7.2 Grounding

G3/G4 的唯一 auxiliary head 为 `Linear(192,1)`：

```text
y_hat_t = H_FTV(r_online_t)
L_FTV = patient-mean SmoothL1(y_hat_t, y_t) over valid visits
L_total = L_base + lambda_FTV * L_FTV
```

所有有效 visit 等权，先患者内平均再对 grounded patients 平均。FTV target 不属于 model forward input；无 FTV 的患者得到可微零 auxiliary loss但继续优化 `L_base`。正式 feature extraction 移除/忽略 head，且不加载真实 FTV。

## 8. Fold-0 lambda pilot 与 checkpoint 选择

候选严格为 `{0.02,0.05,0.1,0.25}`。fold 0 同时评估 G3 与 G4，用已训练 G1/G2 fold-0 validation 指标作为各自配对 baseline；不读取 test feature、test FTV、pCR 或 test AUROC。

硬门槛：

1. loss/gradient finite，跨患者 `r` feature std 不低于 0.05，无 collapse；
2. validation JEPA state loss 相对配对 baseline 恶化不超过 5%；
3. 用 train-only StandardScaler 与固定 alpha=1 Ridge 对四时点 validation FTV 做 macro 评估，至少一个 pairing 的 macro Spearman 增加不低于 0.03，且另一个 pairing 不出现低于 -0.02 的明显反向变化。

通过硬门槛后选最小 lambda；若无候选全部通过，只能在满足稳定性与 5% base 门槛者中按两 pairing validation macro Spearman 平均最高者 fallback，并明确标记 pilot grounding 证据不足。锁定同一个 lambda 后用于 G3/G4 全五折。

每 epoch 记录 raw base/FTV loss、weighted FTV、encoder/head gradient norm、representation std、grounded/ungrounded patient 与 visit 数、learning rate。G1/G2 checkpoint 在非 collapse epoch 中按最低 validation base loss选择；G3/G4 在 validation base 不超过对应 baseline best 的 5% 且不 collapse 的 epoch 中按最低 validation FTV loss选择，fallback 规则写入 selection JSON。任何 test 数据都不参与 epoch 选择。

## 9. Smoke test 与 mask 防泄漏

正式训练前 G1–G4 各运行至少 1 epoch real-cache smoke，覆盖 forward、backward、validation、checkpoint save/load。必须验证：

- DCE tensor 恰为 7 channels，第一层卷积 `in_channels=7`；
- G1/G3 不接受 mask；G2/G4 mask 只进入单一 normalized pooling 函数；
- 固定 DCE、换 mask 时 encoder spatial map bitwise 不变；
- `pool(F,M) == pool(F,cM)`；常量 spatial map 对不同非空 mask volume 输出相同；
- all-ones mask 等于 GAP；empty mask 有限且严格回退 GAP；
- 不同 FTV target 不改变 forward `r/z`；移除 FTV head 不改变 `r/z`；
- G4 的 FTV loss 对 backbone、Linear response projection 与 head 均有非零 gradient，对 EMA 无 gradient；
- FTV patient/timepoint alignment、fold transform hash、train/val/test disjoint；
- 源码与 state schema 中不存在 mask geometry、voxel count、explicit volume 路径。

报告写入 `reports/smoke_test_report.md`。

## 10. 五折训练协议

统一设置：base channels 16、response/projected dim 192、Transformer depth 3、heads 4、MLP dim 512、dropout 0.1、batch size 32、AdamW、learning rate `5e-5`、weight decay `1e-4`、最多 12 epochs、patience 4、gradient clipping 5、seed `2026+fold`。所有 I-SPY2 fold-train patients 加 156 名 I-SPY1 参与 base loss；validation 只用相应 I-SPY2 validation patients。

每个 checkpoint 保存 resolved config、split/hash、transform/hash、architecture contract、shared initialization hash、plan hash、代码 hash、source/current commit、epoch selection evidence。G0 保存 source checkpoint/feature 的路径和 SHA，不复制或覆盖原文件。

## 11. Frozen representation audit

训练结束后冻结 online encoder 与 response projection，每个 model/fold 只提取一次 `[808,4,192]` 的 `r`，供全部 probe 与 pCR 共用。G0 从已有 OSRA NPZ 的 `online_preprojector` 物化标准化引用记录。

统一单输出 Ridge 协议：feature scaler 与 Ridge 只在 outer train 拟合；alpha grid 为 `{1e-4,1e-3,1e-2,1e-1,1,10,100,1000}`，仅按 validation standardized MSE 选择；锁定后 test `predict` 一次。

主分析：

- static：`r_t -> FTV_t`，T0/T1/T2/T3 分开；
- longitudinal：`r_(t+1)-r_t -> ΔFTV`，三个相邻 transition 分开；训练阶段未使用 ΔFTV；
- transfer：static LD/sphericity 与 ΔLD/Δsphericity，完全相同 Ridge protocol。

报告 Spearman、Pearson、R²、MAE、RMSE、train-mean B0 RMSE gain、prediction/target variance ratio；提供 fold 指标、pooled OOF、patient-level bootstrap 95% CI，以及 G3−G1、G4−G2 的 exact-patient paired bootstrap。单 cell CI 标注为未作多重比较校正，且只条件于单训练 seed。

## 12. Image-only pCR readout

严禁 clinical、treatment、radiomics、geometry、mask feature、真实/预测 FTV 或 FTV-head output。每个 decision point 独立使用：

```text
T0:       r0
T0-T1:    concat(r0,r1,r1-r0)
T0-T2:    concat(r0,r1,r2,r1-r0,r2-r1,r2-r0)
```

每 fold/model 在 train 拟合 StandardScaler 与 class-balanced LogisticRegression。validation 选择 penalty `{l1,l2}`、C `{.001,.003,.01,.03,.1,.3,1,3,10}` 与 Youden threshold；test 只预测一次。报告 AUROC、AUPRC、accuracy、sensitivity、specificity、pooled OOF、fold mean±sample SD、patient paired bootstrap。primary comparisons 是 G3−G1 和 G4−G2；G4−G0 只描述不同 input contract，不作为主要成功依据。

## 13. Prediction-level 与可复核产物

Representation prediction CSV 至少含 `patient_id,fold,split,model,timepoint,transition,representation,target,y_true,y_pred`，并补充 task、standardized values、selected alpha、transform/source hash。pCR CSV 至少含 `patient_id,fold,model,decision_point,y_true,probability,predicted_label,threshold`，并补充 split、readout hyperparameters 与 feature source hash。

预期 complete measurement OOF probe 主表为 39,375 行：`5 models * 375 patients * (4 static + 3 change) * 3 targets`；pCR OOF 为 12,120 行：`5*808*3`。所有文件生成 SHA manifest、唯一键与 coverage 检查，patient-level 文件保持本机受控环境，不提交公开 Git。

## 14. 统计比较与预注册成功门槛

所有 primary difference 都按相同患者、相同 fold、相同 cell 成对 bootstrap 2,000 次。

对 G3−G1、G4−G2 分别定义：

- A（static grounding）：四时点 macro ΔSpearman ≥0.05 或 macro ΔR² ≥0.05，且对应 95% CI 下界 >0；另一指标不得显示明确反向证据。
- B（observed ΔFTV）：三 transition macro ΔSpearman 或 ΔR² ≥0.05 且 95% CI 下界 >0，或至少一个主要 transition 达标且其余 transition 没有 ≥0.05 的稳定反向下降。
- C（longitudinal pCR）：T0–T1 与 T0–T2 AUROC 差均为正、两者 macro gain ≥0.02 且 macro paired 95% CI 下界 >0。

同时要求 grounded 模型的 pCR 不出现明确下降、representation 不 collapse、validation base degradation ≤5%。这些阈值用于决策而不是事后按 test 调整。

## 15. GO / PARTIAL GO / NO-GO

- **GO**：G3 或尤其 G4 满足 A，且同时满足 B 或 C；建议下一阶段研究 grounded `r_0:t -> predicted r_(t+1)`，但本任务不实现。
- **PARTIAL GO**：满足 A，但 B/C 均不满足；说明形成 static tumor-burden state，尚未形成可靠 dynamic/pCR response state。
- **NO-GO**：G3/G4 相对配对 baseline 连 A 都不满足，或改善只能由不合格 shortcut/representation collapse解释；不继续给当前小型 3-D CNN 堆 transition complexity。

## 16. 必须生成的图与报告

至少生成 12 张图：G0–G4 static FTV Spearman、static R²、ΔFTV Spearman、ΔFTV R²、G1/G3 paired gain、G2/G4 paired gain、FTV scatter、ΔFTV scatter、三 decision point pCR AUROC、mask contract comparison、LD/sphericity transfer、base/FTV/std training curves。

最终中文报告写入 `reports/final_report.md`，覆盖科学问题、前轮启发、G0–G4、mask 控制、grounding、lambda、static/ΔFTV/transfer/pCR/stability/ablation/paired statistics、限制、决策与下一步，并逐条回答用户要求的 a–i。

## 17. 机器化验收

最终生成 `metrics/acceptance_check.json`，逐项对应 19 条验收标准，并额外核对：G3/G1 与 G4/G2 shared initialization hash、25 个 model×fold 资产 coverage、history 必需列、每个 feature NPZ `[808,4,192]`、prediction 唯一键/行数、paired patient set、12 张图可解码与 SHA、报告相对链接、输入 metric hashes，以及 `metrics/decision.json` 的预注册 gate 计算。
