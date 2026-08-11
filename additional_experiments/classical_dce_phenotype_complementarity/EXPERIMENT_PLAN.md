# Classical DCE Phenotype Complementarity Baseline

## 1. 研究问题与预注册边界

本实验检验传统 DCE measurement phenotype 是否在 FTV 与 pretreatment Patient Profile 之外提供 pCR 预测信息，并检验其对 HR、HER2 与四分类 HR/HER2 subtype 的可解码性。实验只使用真实工作簿中确认存在的字段，不按 held-out test 表现筛选 feature family，不扩展到 RF、XGBoost 或 MLP。

主结论来自 384 名同时具有完整 DCE measurement、clinical profile 与 pCR 的患者，因而不依赖 current MRI latent 是否可用。MRI reference 仅在另外的 375 人完全 matched 子集上报告。

## 2. 数据与真实 feature family

源 measurement 为 `Multi-feature-MRI-NACT-Data.xlsx` 的唯一 sheet `datawith4visits`（384×29）。数据是低维纵向 measurement，而非高维 texture radiomics：

- F：FTV，原始列 `VOLUME_TUM_BLU_V10`–`V40`；
- D：LD，`LD_T0`–`LD_T3`；
- S：sphericity，`SPHERICITY_T0`–`SPHERICITY_T3`；
- B：BPE 5-slice mean，`BPE_5slice_mean_T0`–`T3`；
- NONFTV = D + S + B；FULL = F + NONFTV。

工作簿另外给出 12 个相对 T0 的 percent-change 派生列，不把它们误当成独立观测。模型从绝对值按预注册公式重建变化，便于 timing 检查。

## 3. Timing contract

机器可读契约见 `information_timing_contract.csv`。Static `Tk` 只使用当前 `Tk` 的绝对 measurement。Longitudinal prefix `Tk` 使用 T0…Tk 的绝对 measurement，并对每个已观察随访加入：

`absolute_change(T0,Tk) = x_Tk - x_T0`

`relative_change_pct(T0,Tk) = 100 * (x_Tk - x_T0) / abs(x_T0)`

真实数据四类 T0 baseline 均为有限非零值，故相对变化对 384 人均有定义。T3 始终标记 `late/pre-surgery`。任何 timing 禁止使用未来访视。

## 4. Population、split 与 missingness

Primary 为 clinical-radiomics complete 384。固定 seed 20260811，以 joint `pCR × HR × HER2` 分层建立五个 outer test fold；每个 outer non-test pool 再分层抽取 20% validation。所有 patient row 仅属于 train、validation 或 test 之一。

Primary missingness estimand 是 strict matched complete-case：每个 paired comparison 在同一 timing、view、fold 使用完全相同患者。Secondary 使用 outer-train-only median imputation 与 missingness indicator；当前真实 radiomics 列无缺失，因此它是实现与未来数据稳健性检查，预期与 complete-case population 相同。Clinical numeric median、categorical vocabulary/缺失 level 均只从 outer train 拟合。

## 5. Preprocessing 与模型选择

- FTV、LD、BPE absolute value：逐元素 `log1p`；SPH：identity；
- signed absolute/relative change：identity；
- 每一列 1%/99% winsorization boundary 只在 outer train 拟合；
- median、missingness indicator 与 StandardScaler 只在 outer train 拟合；
- validation 只选择模型超参数与 binary threshold；test 只评估一次。

Patient Profile `C` 固定为 HR、HER2、MP、age、race、menopausal status、ethnicity 与 assigned treatment arm；race/menopausal status 使用与既有 complementarity audit 相同的预先定义语义合并（例如多种族合并为 `Multiple`，格式不同的 pre/peri-menopausal 文本合并），不是按 outcome 学习的类别。Primary 是 L2 logistic regression，`C ∈ {1e-4,1e-3,0.01,0.1,1,10,100}`。Secondary 是 RBF SVM，`C ∈ {0.1,1,10,100}`、`gamma ∈ {0.001,0.01,0.1,1}`；概率校准只使用 outer-train 内部数据。超参数以 validation AUROC 选择，确定性 tie-break 取更强正则/更小参数。

## 6. pCR baselines 与 probe

每个 timing、static/longitudinal view 评估 C、F、N、FULL、C+F、C+N、C+FULL。核心 paired effect 是 C+FULL − C+F；另报告 C+N − C 与 N standalone。pCR 指标为 AUROC、AUPRC、balanced accuracy 与 Brier。

N 与 FULL 分别预测 HR、HER2、四分类 subtype，报告 AUROC、AUPRC 与 balanced accuracy；multiclass 使用 macro one-vs-rest 指标。相同 outer folds 与 timing contract 被复用，不为 probe 重新挑 cohort。

## 7. FTV redundancy 与 residualization

每个 outer fold 内，以 ridge `NONFTV -> current transformed FTV`，validation 选择 alpha，outer test 报 R² 与 Spearman。另在 outer train 以固定 `alpha=1` 的多输出 ridge 拟合 standardized `FTV prefix -> NONFTV`；该固定值不依据 pCR/validation/test 选择，validation/test 只应用冻结映射，并用残差 `NONFTV_res = NONFTV - E_train[NONFTV | FTV]` 训练 N_res 与 C+F+N_res pCR readout。Residualization 从不读取 test outcome。

Family ablation 固定为 C+F+D、C+F+S、C+F+B，不穷举组合，也不按 test 结果选择 family。

## 8. Uncertainty、MRI reference 与科学分类

关键比较 C+F vs C+FULL、C vs C+N、C+F vs C+F+N_res 使用 2,000 次 paired patient-level bootstrap；AUROC/AUPRC 的 delta 为 augmented − baseline，Brier improvement 为 baseline − augmented。重采样保持 paired prediction，并在 `outer-fold × pCR outcome` strata 内抽样，以保持各 held-out fold 与类别构成且避免无双类别的 AUROC draw。

已有 frozen LOCAL MRI OOF predictions 可读时，在 375 名 FTV/MRI 完全 matched 患者上附加 M、C+M、C+F+M，并在同一患者子集比较传统 phenotype。该 reference 不改变 384 人 primary radiomics classification。

预注册科学综合规则：A 要求 C+FULL 相对 C+F 的 AUROC CI 在至少两个 timing（至少一个 T0–T2）完全高于零，且 residualized effect 至少一个 timing CI 高于零；B 要求 standalone N 有 AUROC≥0.60，同时 C+FULL−C+F **没有任何单个 timing 的正 CI** 且 residual signal 完全不存在；C 要求 **N（不是仅 FULL）** 对 HR/HER2/subtype 至少一个 AUROC≥0.60，同时 standalone pCR、任何 incremental 正 CI 与 residual signal 均不成立；D 要求上述 pCR 与 profile signal、任何 incremental 正 CI、residual signal 均不成立。这里不把“未满足稳定 A”偷换成“≈/equivalent”；没有预注册 equivalence margin。若观测组合不满足任何一个原始 A–D 定义（例如只有孤立的 late incremental/residual effect、但没有稳定增量），报告 `MIXED/INCONCLUSIVE`，绝不强行贴上“FTV-redundant”机制标签。阈值只用于保守综合，不替代 effect size、CI 与多 timing 一致性。

## 9. 提交边界

只提交代码、配置、aggregate metrics、图、报告、无直接 patient ID 的 aggregate manifest。Raw workbook、MRI、patient-level split、OOF predictions 与 bootstrap draws 均保持外部或 `.private` 并由 `.gitignore` 排除。
