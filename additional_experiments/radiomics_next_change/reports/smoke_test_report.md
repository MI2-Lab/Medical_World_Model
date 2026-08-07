# M0/M1/M2 Smoke Test 报告

日期：2026-08-06
环境：conda `bowen`；3× NVIDIA RTX PRO 6000 Blackwell Max-Q；分支 `feature/ispy-clean-corejepa`；commit `c413ec86af04795434bdc19e65bbb006c966f379`

## 1. 结论

真实 DCE8 cache 上的 M0、M1 delta-only、M1 delta+state 和 M2 均完成 forward、backward、validation 与 checkpoint 保存，未出现 NaN/Inf 或无效梯度。模型 `forward` 只接收 MRI tensor，不存在 clinical、treatment、9D geometry 或 radiomics 输入参数；M2 的 radiomics head 只读取 predicted image delta。375 名配对患者的 target/mask 与三个相邻 transition 对齐，433 名未配对 I-SPY2 患者及 156 名 I-SPY1 额外预训练患者仍可参加 image loss。

在相同 fold 0、seed 与 24 名 training smoke 子集的 2-epoch 比较中，delta+state 的 validation state loss、delta loss、normalized transition gain 均优于 delta-only，因此它进入正式 12-epoch validation pilot。正式 pilot 仍锁定 delta+state，但也发现 delta-only 的 raw gain 与 delta cosine 更好；完整反向证据见 `reports/m1_variant_pilot.md`。整个变体选择没有加载 test。M2 `lambda_rad=0.1` 的 masked radiomics loss 有限且权重数量级未主导 image objective，允许进入正式 lambda pilot。

## 2. 对齐与泄漏硬检查

- primary cohort：808 名 I-SPY2；额外 image-only pretraining：156 名 complete-four-visit I-SPY1；两者均有只读 DCE8 cache。
- 五折：每折 train/validation/test 无交集，808 人每折恰好出现一次、每人恰好一次进入 test。
- radiomics：375 人；每人 3 个相邻 transition×4 个 feature，合计 4,500 个有效 feature-level mask。
- fold 内 paired training patient 数：247、239、240、242、225。
- 五个 transform 均只由相应 fold 的 I-SPY2 train IDs 拟合；保存 train-ID SHA256、raw-target SHA256 与算法版本。
- transform 定义：FTV/LD 使用 fold-train-only epsilon 的 log-change；sphericity/BPE 使用 absolute change；随后做 train-only 1%/99% winsorization 和 median/IQR scaling。
- 结构检查：M0/M1/M2 的 transition、encoder 与 readout 路径均不读取 pCR；pCR 只留给冻结 readout。

机器可复核结果位于 `metrics/implementation_validation.json` 与 `configs/radiomics_transform_fold_0.json` 至 `fold_4.json`。

## 3. 小张量契约测试

对四种 mode 各使用 `[B=2,V=4,C=8,Z=8,Y=16,X=16]` 的随机 tensor：

| Mode | target state | predicted next | frozen readout feature | backward | M2 head |
|---|---|---|---|---|---|
| M0 | `[2,4,16]` | `[2,3,16]` | `[2,48]` | 通过 | 不实例化 |
| M1 delta-only | `[2,4,16]` | `[2,3,16]` | `[2,48]` | 通过 | 不实例化 |
| M1 delta+state | `[2,4,16]` | `[2,3,16]` | `[2,48]` | 通过 | 不实例化 |
| M2 | `[2,4,16]` | `[2,3,16]` | `[2,48]` | 通过 | 输出 `[2,3,4]` |

无 radiomics 的样本将 `mask=0`，radiomics loss 为零但 image encoder/transition 仍有梯度。M2 的 `radiomics_prediction` 与 target/mask shape 完全一致。

## 4. 真实 DCE8 训练 smoke

共同设置：fold 0；24 名 I-SPY2 training patients；validation 前 12 名；batch size 8；2 epoch；EMA momentum 0.996；learning rate `5e-5`。这是接口/数值稳定性检查，不作为效应估计。

| Mode | epoch 2 val state loss | val delta loss | val radiomics loss | val transition gain | val feature/global state std | 状态 |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0.7433 | 0.03962 | 0 | -23.987 | 0.3809（旧 global 诊断） | 通过 |
| M1 delta-only | 0.1638 | 0.01123 | 0 | -5.575 | 0.2386（旧 global 诊断） | 通过 |
| M1 delta+state | 0.09215 | 0.00859 | 0 | -4.042 | 0.3265（旧 global 诊断） | 通过、进入正式 pilot |
| M2，`lambda_rad=0.1` | 0.09028 | 0.00843 | 0.55331 | -3.959 | 0.3298（旧 global 诊断） | 通过 |

Smoke 中 gain 仍为负，说明两轮更新不足以击败强 copy-current baseline；它不是正式结果，也不是阻塞。正式训练将报告 raw/normalized copy error、稳定的 aggregate gain、per-transition 分布和 delta cosine，而不会只依赖该均值。

## 5. 正式 batch 与资源检查

另以真实 DCE8、32 名 training patients、batch size 32 运行 M0 单 batch，forward/backward 与 checkpoint 保存成功。正式配置因此使用与 clean 预算一致的 batch size 32。正式 M0 首轮实测单 fold 每个 train epoch 约 14–35 秒、validation 约 2–3 秒；PyTorch allocated peak 约 28.7 GiB，GPU 驱动侧总占用约 74 GiB，低于单卡约 97 GiB。

Transformer 发出 `norm_first=True` 导致 nested-tensor fast path 未启用的 warning；它不改变数值或因果 mask，未造成中断。

## 6. Checkpoint 与选择规则

Smoke checkpoint 位于独立 `checkpoints/smoke_*` 目录，不与正式 run 共用。正式 best epoch 的 eligibility 使用跨患者 feature-wise latent std；best metric 不含 pCR。M0 监控 validation state loss；正式 M1/M2 监控仅由影像产生的 raw aggregate transition gain，并同步报告 normalized gain。M2 radiomics loss与随机 SIGReg 不进入 best-epoch 排序，避免由 paired subset 或随机投影决定全部患者的训练长度。

正式 run 默认以原子 claim 拒绝同名并发或覆盖；checkpoint 内保存 fold manifest SHA256、split patient hash、radiomics transform SHA256、Git 信息和模型输入契约。

## 7. 后续执行状态

- M0、M1 正式五折及其冻结 image-only readout/shortcut 已完成。
- 正式 M1 variant 选择只使用 fold 0 validation，锁定 delta+state；见 `reports/m1_variant_pilot.md`。
- M2 的 `lambda_rad={0.05,0.1,0.25,0.5}` 继续严格只使用 fold 0 validation；锁定后才允许启动 M2 五折 test 评估。
- 主五折完成后统一运行 paired radiomics grounding、C0/C1/C2、bootstrap 聚合与作图。
