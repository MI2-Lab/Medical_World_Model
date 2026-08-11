# Model execution ledger

更新日期：2026-08-11。状态字段只描述已真实完成的动作；未完成项不写成成功。

## Environment

| Item | Value |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.9.1+cu130 |
| CUDA runtime | 13.0 |
| GPU | 3 × NVIDIA RTX PRO 6000 Blackwell Max-Q，约 98 GiB/卡 |
| NumPy / pandas | 2.2.6 / 2.3.3 |
| scikit-learn | 1.8.0 |
| SciPy / matplotlib | 1.16.3 / 3.10.8（均已在 `requirements.txt` 精确固定） |
| joblib / threadpoolctl | 1.5.3 / 3.6.0（scikit-learn 正式运行时传递依赖） |
| timm | 1.0.22 |
| transformers | 5.2.0（正式 DINO forward 不依赖 Transformers） |
| gdown | 5.2.0 |
| pytest | 8.4.2 |
| Extraction precision | CUDA bf16 autocast，输出 float32 |
| Determinism | `torch.use_deterministic_algorithms(True)`；`CUBLAS_WORKSPACE_CONFIG=:4096:8` |

## Data and evaluation provenance

| Contract/artifact | Population / role | SHA-256 |
|---|---|---|
| Formal five-fold manifest | I-SPY2 808 人，275 pCR+；每人恰好一次 outer-test | `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` |
| Clinical labels | 808 人；HR/HER2/MammaPrint/age/exact arm + pCR | `b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436` |
| C1B Stage-B cache manifest | 947 人 cache contract，其中 808 I-SPY2 进入 probe | `672ad7436b19f30a89640a2b36504f1e7fbaaff83fd07bc058c008b204d2a3c9` |
| Radiomics/FTV transition table | 375 complete cases，110 pCR+；paired sensitivity only | `26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d` |
| Model/input lock | 正式模型、checkpoint、adapter 与 label-blind extraction code | `639ec6e82f67c3927d8f543dfa3f9834ac8e09aec4820906c138b1839662c377` |
| Formal evaluation lock v1 | 21 个 code/config/test files + 所有正式输入资产；保留失败历史 | `45969bee8a0cd40466b6538f80b2b6fc35dd4bbf6abba7f021144e9c8489f272` |
| Formal evaluation lock v2 | baseline-v2 producer 与 subtype OVR 修订 | `b15f7023b7021f5c1169b51cf6bc8fe0cc1d9085102a61fbdb1d68589fe2edc5` |
| Formal evaluation lock v3 | probe-v3 producer、Ridge 数值门与 mixed-lineage contract | `8e8c4a5488fc862e1c73ac643495216bcc6eb015b767d7be39150894c8265104` |
| Reporting lock v1 | 只修改 binary IRLS public/recompute identity tolerance | `5be484747ac6aeb5b622ecb70f2590e214f687e714e6d111a73f4ee775b165ca` |
| Finalization lock v1 | outcome 查看前冻结的报告 schema、完整性门与结论规则 | `30c0c0e6ce7d92fb6164368addf467e2142e5159fb95240f4c030ddb986c4e7b` |
| Static comparison contract | 5,000 次 paired bootstrap，seed 2026 | `f99fd76bd35b784500194347c4b363725616ca6adab2ba830a006cc7cc4a7e13` |

C1B-H 单人张量固定为 `float32 [4,7,112,176,160]`；DCE7 依次为 `pre, early, late, early-pre, late-pre, peak-relative-enhancement, late-minus-peak-relative-enhancement`，所有 channel 已按 patient/visit 在 valid-source 区域 robust normalize 并截断至 `[-5,5]`。四个 visit 共用 T0 anchor/grid，spacing 为 `(0.9,0.9,2.0)` mm。

- MedicalNet：七个 DCE channel 逐通道送入同一 frozen 3-D encoder；native layer4 为 `[7,2048,14,22,20]`。GLOBAL 为空间均值，LOCAL 为同一 map 上固定中心 64-mm sampling-cell overlap pooling；每 visit `14,336` 维。
- DINO v1：固定 DCE index `[1,2,4]`，即 `early, late, late-minus-pre`；GLOBAL 为完整 FOV 的 32 张均匀 axial slices，LOCAL 为 64-mm cube 采样到 `32x72x72`；二者均 bicubic 到 224、clip `[-5,5]`、映射至 `[0,1]` 后作 ImageNet normalization。每 slice 拼接 final CLS 与 mean patch token，再对 slice 均值；每 visit `1,536` 维。
- LOCAL 不使用 lesion mask、FTV、clinical 或 outcome，但继承冻结的 T0 localisation prior。pCR 只运行 `T0=z0`、`T0-T1=[z0,z1,z1-z0]`、`T0-T2=[z0,z1,z2,z1-z0,z2-z1,z2-z0]`；不把 T3 用于早期预测。

可发布 artifacts 分别来自 baseline-v2 与 probe-v3 各一次成功、无 `--overwrite` 的 publication；v1/probe-v2 失败 attempts 作为不可覆盖历史另行保留。共同环境为 `PYTHONHASHSEED=2026`、`FOUNDATION_MRI_SELECTION_WORKERS=1`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。输入为上述两个 foundation private NPZ 与 GAP0/LOCAL0 seed-2026 五折 path template；baseline-v2 与 probe-v3 的 exact argv 分别由对应 protocol、lock 与 metric-free receipt 绑定，reporting retry 的 empty argv 由 `REPORTING_LOCK.v1.json` 绑定。

## Acquisition and load gates

| Model/artifact | Official source | SHA-256 | Load gate | Status |
|---|---|---|---|---|
| MedicalNet archive | Tencent README 的 official GDrive ID `13tn…` | `4ba2ece5f32a13b166e431b78e99052c9142f879f60a150baa54ba5068eaf84b` | archive hash；只提取指定 member | PASS（本地，ignored） |
| MedicalNet 3DSeg-8 ResNet-50 | archive member `pretrain/resnet_50.pth` | `5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84` | 318/318 entries、46,155,072 params、strict=True、无 decoder/FC | PASS |
| DINO v1 ViT-B/16 | `dl.fbaipublicfiles.com/dino/...` | `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e` | 150/150 tensors、85,798,656 params、native eps=1e-6、strict=True | PASS |

MedicalNet 发布方未提供 archive/checkpoint checksum，因此其 artifact gate 是 acquisition-time conditional pass；本地 hash 已被冻结，任何 byte drift 均 fail-closed。

## Input/representation smoke

在不读取任何标签或 test metric 的情况下，只对 canonical order 的首个 C1B patient 做 label-blind extraction smoke：

| Model | Input gate | Output | Finite | Private mode | Status |
|---|---|---|---|---|---|
| MedicalNet | 4×7×112×176×160；layer4 每 visit 7×2048×14×22×20 | 4×2×14,336 | 是 | 600 | PASS；首次因缺少 cuBLAS deterministic workspace 硬失败，修复后重跑通过 |
| DINO v1 | 4 visits × GLOBAL/LOCAL × 32 slices × 3×224×224 | 4×2×1,536 | 是 | 600 | PASS |

首次 MedicalNet smoke 的真实错误为：严格 deterministic CUDA 拒绝在未设置 `CUBLAS_WORKSPACE_CONFIG` 时执行 cuBLAS。该次没有 feature shard；代码随后在首次 CUDA handle 前固定 `:4096:8`，未关闭 deterministic gate。

## Candidate assets not executed formally

- BiomedCLIP exact-revision snapshot：候选阶段下载并检查 config/state dict，因污染 gate 拒绝；未处理 patient image。
- DINOv3：候选阶段只用已有本地 cache 验证可加载性；正式实验目录中的匿名下载因 gated 401 失败；因语料/许可/access gate 拒绝。
- DINO v1 HF conversion：为来源核验下载 exact revision；正式 forward 改用 Meta 原生 checkpoint，避免 LayerNorm epsilon 与 pooler 歧义。
- MedicalNet `resnet_50_23dataset.pth`：随 official archive 获得但因预训练数据清单不透明拒绝。

以上 checkpoint/cache 均被本实验 `.gitignore` 排除；不会提交或 push。

## Formal runs

| Stage | MedicalNet | DINO v1 | Notes |
|---|---|---|---|
| 808-patient frozen feature extraction | PASS，536.69 s | PASS，411.37 s | 两个进程分别固定在 CUDA:0/CUDA:1；提取期间不加载 clinical、FTV 或 outcome |
| 808-patient pCR OOF evaluation | v1 exit 143；baseline-v2 PASS | v1 exit 143；baseline-v2 PASS | 630/630 folds、11,340/11,340 candidates；public 252 rows |
| HR/HER2/subtype probes | v1 exit 143；probe-v2 exit 1；probe-v3 PASS | v1 exit 143；probe-v2 exit 1；probe-v3 PASS | v3：300/300 folds、3,300/3,300 candidates、2,160/2,160 OVR estimators |
| FTV/ΔFTV decodability | probe-v2 exit 1；probe-v3 PASS | probe-v2 exit 1；probe-v3 PASS | 375 complete-case paired sensitivity；public 84 rows |
| Beyond-FTV pCR | baseline-v2 PASS | baseline-v2 PASS | 同一 375 人、两 spatial axes、三个 timing；12 个固定 paired comparisons |

正式 feature integrity（患者级文件均为 ignored、mode 600，不进入 Git）：

| Model | Shape / dtype | Feature SHA-256 | Extraction-contract SHA-256 | Coverage / finite |
|---|---|---|---|---|
| MedicalNet | `[808,4,2,14336]` / float32 | `ca45a46bd62e18e42b6d3f2426ce4690a4f3dbf7c2f44804ab0d19bd333ee4a2` | `6f48556e959a4e7984935a99773493906a994173655cab7d38da8ad841d72f3a` | 808 unique、canonical-order digest 匹配、全 finite |
| DINO v1 | `[808,4,2,1536]` / float32 | `c078cd4ddc0c745c32ebcca247d44ef8025d08495f6e3193a481563f0d53ffbc` | `f4e102233ec30d2ae7b8a0d3ac87918052ac08b73cc483a99f1c5621853143d5` | 808 unique、canonical-order digest 匹配、全 finite |

MedicalNet 的 frozen ReLU layer4 有 8,435/14,336 个跨全部 visit/patient/spatial 恒定维；这是正式提取后、仍未读取 outcome 时做的 label-blind representation QA，未据此修改 adapter 或筛选模型。`StandardScaler` 对恒定列使用 scale 1，后续所有 formal candidates 保持原始冻结维度并完整报告。

### Probe v1 computational abort（保留，不覆盖）

正式 `run_probes.py` v1 在原 evaluation lock SHA `45969bee8a0cd40466b6538f80b2b6fc35dd4bbf6abba7f021144e9c8489f272` 下启动，环境为单 selection worker 与单 BLAS/OpenMP thread。约 39 与 61 分钟时，scikit-learn 分别发出一次：

`ConvergenceWarning: The max_iter was reached which means the coef_ did not converge`

只读代码/运行时审计随后确认：四分类 subtype 路径共有 6 sources × 5 folds × 18 L1/L2-C candidates = 540 个 dense multinomial SAGA fits；v1 使用 `tol=1e-7`、`max_iter=20,000`，前两个触发 warning 的 MedicalNet fits 均已耗尽 20,000 iterations，按实测速率为多日级，并且 v1 会让未收敛 candidate 进入 validation selection。这既不具现实可执行性，也不能作为可靠的线性 probe。

因此 operator 在约 65 分钟时仅终止 probe v1：首次 Ctrl-C 因底层 fit 未响应，随后对精确 PID 发送 SIGTERM，进程 exit code 143。baseline v1 没有被中止。终止检查确认 `metrics/` 与 `predictions/` 仍只有 `.gitkeep`；v1 的 atomic writer 尚未运行，没有 patient-level 或 aggregate 输出可复用或查看。

必须准确限定 outcome 边界：v1 已加载标签，且内部可能已经完成部分 HR/HER2 outer-test prediction；因此不能声称“未运行 test evaluation”。可以且经审计确认的是：中止前没有打开、打印、发布或查看任何 prediction、validation metric 或 test metric；修订只由 wall-time 与两条 max-iteration warning 触发，不受模型表现影响。v1 lock 必须按 bytes 保留，后续 probe-v2 lock 以其 SHA 链接；不得把 v1 失败静默改写成一次成功运行。

### Baseline v1 computational abort（保留，不覆盖）

同一 v1 lock 下的 `run_baselines.py` 包含 126 个 model×timing CV results、630 个 fold selections 与 11,340 个 liblinear candidate fits。该进程持续单核计算且没有发出 `ConvergenceWarning`；但 v1 代码同样只记录 `n_iter`、不把 `n_iter >= max_iter` 提升为 hard failure。只读运行审计因此在未查看任何 metric 的情况下预先声明：继续运行至 150 分钟或首条 convergence warning，以先到者为 operator boundary；若完成则要求 post-run 11,340/11,340 candidates 全部严格低于 20,000 iterations。

进程在 150 分钟时仍未退出，且 atomic writer 尚未运行。operator 按预声明边界向精确 PID 发送 SIGTERM；exit code 143。终止后再次确认 `metrics/` 与 `predictions/` 没有 `.gitkeep` 以外的文件。因此 baseline v1 没有可复用/查看的 prediction、selection 或 aggregate metric。与 probe v1 相同，baseline v1 已加载 outcome 并在内存中完成部分 outer-test prediction，故准确表述是“无 outcome-derived result 被查看并影响 v2”，而不是“从未处理 outcome”。

v2 修订由两类纯计算/数值完整性事实触发：probe v1 的两次明确 max-iteration failure，以及 baseline v1 在预声明 wall-time 边界仍未完成且缺少 candidate convergence gate。第一版 outcome-blind v2 smoke 保留 multinomial SAGA、仅把 subtype tolerance 改为 `1e-4`；它只读取 formal MedicalNet feature NPZ，并使用 synthetic 四分类标签和 synthetic split，不读取 clinical/fold/radiomics。该 smoke 在预声明 30 分钟 runtime boundary 仍未完成，operator 于 elapsed 30:19 向精确 PID 发送 SIGTERM，exit code 143，且没有 PASS receipt、prediction 或 metric 被写入/查看。

因此最终 v2 保留 binary liblinear `tol=1e-7`，将 subtype 实现冻结为显式四分类 one-vs-rest liblinear、`tol=1e-4`：每个 hyperparameter candidate 拟合四个 class-balanced binary estimators，四个 sigmoid probability 按行归一化后计算原先冻结的 validation macro-OVR metrics。每个 binary 或 subtype 底层 estimator 都必须在任何 validation prediction/metric 之前满足 solver-accurate 的 `0 <= n_iter < max_iter`，且任何 `ConvergenceWarning` 都提升为 hard failure；liblinear 的 `n_iter=0` 只表示强 L1 下初始全零参数已经满足停止条件，同时仍要求参数与概率 finite。该 outcome-blind runtime 修订不改变模型、C grid、intercept scaling、fold、timing、selection rule、test single-use guard 或比较矩阵。

最终 outcome-blind MedicalNet synthetic smoke 覆盖 GLOBAL/LOCAL、完整 36 个 L1/L2-C candidates 与 144 个底层 OVR estimators，并对每个 spatial 重复固定 L2/C=0.1 determinism sentinel。结果为 PASS：总耗时 118.633 秒，观测 `n_iter` 范围 0–19，无 `ConvergenceWarning`；receipt SHA-256 为 `b1a92d4b004a2660cf8c83746b936265e79f422abf06dc2b1bfd2c8adb5fe01f`。Receipt 明确记录 formal manifest、clinical outcome 与 radiomics 均未读取。

### Baseline v2 successful run

`run_baselines.py` 在 v2 lock 下 exit 0。Metric-free receipt SHA 为 `8fb79e1e4316e2fab94e386c3e2a0bbebde074d78771341188818e2d9702f42c`；其 binary-stream 验证同时锁定 argv SHA `755957f2143829fb27dc2317e35d1490dd2d2d630ff98ce06a4ad8691a8a1864` 与下列四个 artifacts。状态机完整结束于 `formal_artifacts_written`：630/630 folds、11,340/11,340 candidates，每 fold 恰 18 candidates；`n_iter` 范围 0–1,485，0 个达到 `max_iter=20,000`，没有 open state 或 convergence failure。

| Artifact role | Visibility / rows | SHA-256 |
|---|---|---|
| baseline aggregate metrics | public，252 data rows | `9ba743cfb6515784bd06d1384c1fe3bcad1d2d9873e00ba438bed1e31b71f52e` |
| baseline OOF predictions | private，64,137 data rows，mode 600 | `d45b55341fc8c6c0c169d547a3b47d81dc55d9423e0e06be0be72bea654b350e` |
| baseline selections | private，630 data rows，mode 600 | `313a7b38e413381adeceee3fbe40f0baf0cd086377346e83de8620b1e010b937` |
| baseline progress | private，23,942 JSONL events，mode 600 | `4e52fe3f379f7e1f25b5892a4ea2d66cd3f00657e058c2ec282c100e7fb1180a` |

### Probe v2 failure and probe v3 successful run

正式 probe-v2 在 v2 lock 下以 exit code 1 失败。最后一个持久化状态是 static-FTV Ridge fold 3 的 selection 完成；原 traceback 没有保留。完全 synthetic、outcome-blind 复现与该日志窗口高度匹配于有限 transformed prediction 经 `expm1` 溢出后触发 outer-test nonfinite gate，但这只是最可能根因，不能写成已恢复原异常。该次已经加载标签并可能在内存处理部分 test prediction；可以确认的是没有 prediction、selection、public metric 或 receipt artifact 被写出或查看，v3 修订不依据模型性能。

v3 对 static FTV Ridge 将 validation 与 outer-test prediction 同样限制到 outer-train transformed-domain min/max；raw transformed nonfinite 仍硬失败，所有 alpha 都增加 warning、iteration 与 finite-parameter gate。正式 probe-v3 exit 0；metric-free receipt SHA 为 `c7b515a099a5da6fb2a88e25cad92ddb6aaa45d2fb8b9a79e7357b4381955505`，argv SHA 为 `0d1af3956391f7c58a48eaffae914c0af51b1eceffb5ba792624d2e04a4923ec`。状态机完整结束于 `formal_artifacts_written`：300/300 folds、3,300/3,300 candidates、2,160/2,160 OVR estimators；binary、subtype 与 Ridge 的最大 `n_iter` 分别为 677、197、805，无 warning、nonfinite 或 capped candidate。

| Artifact role | Visibility / rows | SHA-256 |
|---|---|---|
| phenotype aggregate metrics | public，24 data rows | `042258b147c6076083846aad80c9df71c0e1236b51573410b4025505cd5c93f0` |
| phenotype OOF predictions | private，9,696 data rows，mode 600 | `356705b0e3e21988579d0b1f30242741f57c3d6a42619a71b6da4928822aa877` |
| phenotype selections | private，60 data rows，mode 600 | `f5107a2c295b67a15652c3243a6bfc86539b9665e20686daa7f934b28e13ce85` |
| subtype aggregate metrics | public，12 data rows | `c05256803c629122625078d4f1c9f7eed0974554fcf86720e4d2e17f24162279` |
| subtype OOF predictions | private，4,848 data rows，mode 600 | `f80b571dcf121729910e14290cb47d73ddd3e7f5b928f111d48ab8263a818010` |
| subtype selections | private，30 data rows，mode 600 | `4f949691bb6b3b64cda0fb10b7b2da613bc188fb419edd8ae7f5e357e29b916d` |
| FTV aggregate metrics | public，84 data rows | `0341cffefe45e5e393ff9dcad147c5839331bf32608cd8ade3b2ca49d4bb76a8` |
| FTV OOF predictions | private，15,750 data rows，mode 600 | `36b9b29d51cdde788cdf81193ce670906bf7231e08edc65e4f373c6e1f042f69` |
| FTV selections | private，210 data rows，mode 600 | `980dc29507f532d3088226f8b6fc7c3383e49bfde5365e0cc399294e49a7aaaf` |
| probe progress | private，11,822 JSONL events，mode 600 | `07c21dea7e06ecf8805809ad5fe13cffa1608fdee0e191d50d63454468a078eb` |

### Reporting-only retry v1（不改 evaluation producer）

正式 probe-v3 已在 `EVALUATION_LOCK.v3.json` SHA `8e8c4a5488fc862e1c73ac643495216bcc6eb015b767d7be39150894c8265104` 下 exit 0；metric-free receipt SHA 为 `c7b515a099a5da6fb2a88e25cad92ddb6aaa45d2fb8b9a79e7357b4381955505`。其 10 个 prediction/selection/public aggregate/progress artifacts 由 binary-stream SHA 绑定，审计计数为 300 个完成 folds、3,300 个完成 candidates、2,160 个完成 OVR estimators，0 个 capped candidate。该状态审计没有读取性能值。

首次正式 empty-argv summarizer 在解析完输入、核对 public aggregate 与由 serialized private OOF 重算值时 exit 1；五个 public reporting outputs 与 `reporting_run_provenance.json` marker 均未生成。唯一查看的 numeric identity 是 `pCR / dino_vitb16_imagenet1k_mri_clinical_ftv / GLOBAL / T0-T2 / radiomics_complete_case_375 / outer_fold_macro / calibration_intercept`：既存 public 值 `-0.2992681677870132`，重算值 `-0.29926816783893295`，绝对差 `5.191974628004914e-11`。AUROC、AUPRC、Brier、ECE、prediction、selection 或模型方向/性能值均未查看，修订也未依据这些值。

修订只作用 reporting identity check：原 `rtol=1e-10, atol=1e-12` 对 baseline/phenotype binary IRLS 的 `calibration_slope` 与 `calibration_intercept` 改用 `rtol=1e-8, atol=1e-9`，依据冻结 IRLS `1e-9` step stopping threshold；其他 binary metrics、subtype 全部列、FTV 全部列（包括同名 calibration 列）仍使用原严格阈值。模型、fold、selection、prediction 和 producer artifacts 均不重跑或覆盖。

独立 reporting lock 为 `configs/REPORTING_LOCK.v1.json`，SHA `5be484747ac6aeb5b622ecb70f2590e214f687e714e6d111a73f4ee775b165ca`，parent 为上述 v3 lock，正式 summarizer argv 仍严格为空。冻结 finalizer marker 中 `summarizer.protocol_version="v3"` 是不可变 schema/overall probe-generation label；实际 retry code identity 由 marker 的 `code_lock_sha256` 指向该 reporting lock。`FINALIZATION_LOCK.v1.json` SHA 继续为 `30c0c0e6ce7d92fb6164368addf467e2142e5159fb95240f4c030ddb986c4e7b`，未修改。

唯一正式 reporting retry 随后以 empty argv exit 0；empty argv SHA 为 `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`。它写入五个 public outputs 后最后发布 schema `foundation_mri_reporting_run_provenance_v1` 的 commit marker `reporting_run_provenance.json`（SHA `ea917e3adfb8391fc396d463eb6aa1ed1bd73e0f7182fe0ebcb0a301c487e325`）。五个 outputs 与 marker 均为 mode 0644；marker 绑定 baseline-v2、probe-v3、reporting lock、finalization lock 与全部公开输出：

| Public reporting artifact | SHA-256 |
|---|---|
| `metrics/paired_bootstrap_comparisons.csv` | `76a55e828c024513d5fcbe5ed9db4e2c3768d1d5657fe71cea13be316c08c363` |
| `metrics/results_summary.json` | `bec3af0bd2b73695f5933cce25fb17b8dd2c72589be3784196e5e5f9605575b8` |
| `reports/results_summary.md` | `646d21d815bef09695230a0a4ae6dd153604c810b6f23abbe0fbd69ba294697d` |
| `figures/pcr_timing_performance.png` | `0c0accec1b7a9fda27a59cfedb57870346afa456384f8a6c8357b52a62192df0` |
| `figures/calibration_clinical_complementarity.png` | `31c3e7ea14ba00a6ed862049fdb48b3115c50be7da192eb91036d8a1a70648b5` |

汇总完整覆盖 126 个 pooled pCR cells、132 个预注册 paired comparisons（396 metric rows）、12 个 phenotype cells、6 个 subtype cells 与 42 个 FTV/ΔFTV cells；没有按 test 结果过滤正式候选。

## Known reproducibility limitations

- MedicalNet 发布方没有为 official archive/checkpoint 发布 checksum；本实验只能在取得时从 official archive 锁定 bytes 与 strict state-dict schema，属于 conditional pass。
- 数值与绘图的直接依赖已在 `requirements.txt` 固定；scikit-learn 的传递运行依赖 joblib 1.5.3 与 threadpoolctl 3.6.0 另在本 ledger 记录，但未重复列作 direct requirements。
- `EVALUATION_LOCK.json` 记录了本 site 的绝对 Python 路径，计算合同可审计但不完全可移植；其他 site 应保留相同 package/version/thread contract 并重新生成 site-specific acquisition manifest。
- 正式公开输出先在共同 staging 目录完整构建，再以 hard-link no-overwrite 逐文件发布；异常路径回滚已发布 links，并以最后发布的 provenance marker 作为集合完成标记。跨文件仍不是 filesystem transaction，marker 缺失或任一 hash/schema 不符时 finalizer 必须 fail closed。
- 正式结论仅来自 frozen encoder + linear/Ridge probe；未运行 light/full fine-tuning，也没有外部 cohort。375 人 FTV/radiomics complete-case 缺失非随机，其绝对指标不得与 808 人 primary 混为同一 estimand。
