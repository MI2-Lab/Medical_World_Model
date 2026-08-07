# Observed-State Radiomics Audit 资产检查

检查日期：2026-08-07

## 1. 运行状态

| 项目 | 实际值 |
|---|---|
| Git 分支 | `feature/ispy-clean-corejepa` |
| 起始 commit | `16610447c3752f0943d31f389135c75d1f26350e` |
| 工作区既有未跟踪内容 | `shortcut_audit/`；不读取、不移动、不覆盖 |
| Python | 3.11.14 |
| PyTorch | 2.9.1+cu130 |
| CUDA runtime | 13.0 |
| GPU | 3× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，单卡约 97,887 MiB |
| conda 环境 | `bowen` |

本实验只读取上一轮资产，所有新文件均写入 `additional_experiments/observed_state_radiomics_audit/`。不重新训练 world model，也不改 transition 或 radiomics auxiliary loss。

## 2. 正式 checkpoint

全部 checkpoint 位于 `additional_experiments/radiomics_next_change/checkpoints/<run>/fold_<k>/best.pt`，均可用 `weights_only=True` 安全读取；每个 checkpoint 锁定本折 train/validation/test patient hash、fold manifest、MRI cache、radiomics transform、模型配置和训练实现哈希。

| 模型 | Fold | Best epoch | Checkpoint SHA-256 |
|---|---:|---:|---|
| M0 `m0_final` | 0 | 3 | `3d1ee55defd7dcd0306cf673aac64c39f6484518df77bd0abe6d4771cbad40d3` |
| M0 `m0_final` | 1 | 3 | `5c4ae970c3705d0b0d12b8f2133ef9c67649660cae2d5d3c5b14212e8e47d290` |
| M0 `m0_final` | 2 | 2 | `51a47dbae1321d24d0081c427eb67c469ccc5101b97c10e46604b595d44b78c0` |
| M0 `m0_final` | 3 | 2 | `d759bfbf68c704ce8d669e561900986349d1df15e94683d65ec35aa1dba25d15` |
| M0 `m0_final` | 4 | 3 | `64ba9c50ff81a50dd52b9f431cf0f48c2b2eaebc3c531f8dba3aa4763913eeb0` |
| M1 `m1_final` | 0 | 12 | `c9b4f3a854c33bd7283e931912696e179fc6e466b2a4c91c905009825170d20d` |
| M1 `m1_final` | 1 | 12 | `f0062e6fa541b4ba418bceda97812a500dcd700a75d3e76d0be6a507f37a72c5` |
| M1 `m1_final` | 2 | 12 | `215750ef93c2d0534cbf0e1b49f9b5c6acd5ecac34770dc6509f84577769fab2` |
| M1 `m1_final` | 3 | 12 | `ed90cce02633fcbb201d4d11547f4015438412d7b76920312907d33a8aa4ec98` |
| M1 `m1_final` | 4 | 11 | `73f53abb035b96f2ecbbd3a3dcac3a12f73169a8a6a0a7ba5ba95dc90e000f8a` |
| M2 `m2_final` | 0 | 12 | `aa1fa4dc1a5c64b891b8f80c5e9c59d190962e655486079180e4601b472faef4` |
| M2 `m2_final` | 1 | 12 | `67ff6c9180bf1706bc3efac7ecc009adbfa28b8eb771cc44c2c98e6759b9a0b3` |
| M2 `m2_final` | 2 | 12 | `1dde5b82d4152c160e6c076d47e9a04b4fe3e9c13cca133d66357e5fa9768264` |
| M2 `m2_final` | 3 | 12 | `176bb637185070dc6dc1d14e940dfa4bd49886fbe8238ddc12290d27e274f18d` |
| M2 `m2_final` | 4 | 11 | `039281eef918fa95a461da75bedf1bbd567d5e1cf4e38ba09f944595f06d0512` |

M0/M1 的 `lambda_rad=0`；M2 的正式权重为 `lambda_rad=0.05`。所有模型均为 8 通道输入，且没有独立 clinical、treatment、9-D geometry descriptor 或 radiomics 输入；第 8 通道是二值 ROI mask，因此必须称为“ROI辅助 observed image representation”。

## 3. Fold、患者和数据

- Fold manifest：`$DATA_ROOT/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv`。
- Manifest SHA-256：`143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。
- 808 名 I-SPY2 患者在每折各出现一次，每名患者恰好一次进入 test。fold 0–2 为 525/121/162，fold 3–4 为 526/121/161（train/validation/test）。该 manifest 是经过验证的候选副本，但缺少原生 clean 生成 provenance，不能称为 native clean fold reproduction。
- MRI cache：`$DATA_ROOT/Preprocessed/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96`，964 个 NPZ，约 33.9 GiB；其中 808 名 I-SPY2、156 名仅用于原 world-model pretraining 的 I-SPY1。本 audit 的 probe 只使用 808 名 I-SPY2。
- 每个 NPZ 的 `x` 为 `[4,8,32,96,96]`，顺序为 T0/T1/T2/T3；前 7 通道为 DCE 图像，第 8 通道为二值 ROI mask。
- 全 808 cohort 的 3,232 个 patient-visits 中有 41 个空 ROI mask，涉及 33 人；375 名 radiomics 配对患者中有 12 个空 mask，涉及 8 人。ROI probe 对这些 cell 显式标为无效，不以零向量替代。
- Radiomics overlap：808 行、375 名配对患者，SHA-256 `91b575c9e7e351312b8181a091bdffd2d1f61b88b5a98ac3d78d54c94b63da6b`。
- Source target row table：375×3=1,125 行，文件 SHA-256 `26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d`；FTV、LD、sphericity、BPE 的四访绝对值和三个相邻 transition 均完整。Transform/probe 对同一内容使用 canonical raw-radiomics mapping hash `512749ccf986de4af4c0109b4ce060c61a90112816895a2ae7423784ea60de4e`；两者是不同序列化对象的 hash，不能互换。
- 每折配对 train/validation/test 人数分别为 247/59/69、239/69/67、240/52/83、242/61/72、225/66/84。

## 4. 实际 encoder 计算图和 tensor shape

实际实现位于 `additional_experiments/radiomics_next_change/src/rnc/model.py`：

```text
DCE7 + ROI mask                  [B, 8, 32, 96, 96]
→ 4 个 3-D residual stages
→ 最后一层空间 feature map       [B, 128, 4, 12, 12]
→ AdaptiveAvgPool3d(1)
→ global pooled feature          [B, 128]
→ Linear(128,192) + LayerNorm
→ pre-projector appearance       [B, 192]
→ 2-layer projector
→ projected appearance latent    [B, 192]
```

实测 M0 fold 0 的 online 与 EMA target 两条分支 shape 完全相同，手动中间层计算与公开 `encode_*` 输出的最大绝对差为 0。该 wrapper 根本没有 geometry projector，所以 192-D state 是 appearance 路径；但输入 mask 仍提供 ROI 几何信息。

Transition 读取 online projected latent；上一轮 frozen readout 的 `current_state` 使用 EMA target projected latent。因此本 audit 以 **online** representation 作为“被训练且被 transition 读取”的主分析，以 **EMA target** 作为 readout/target-coordinate 敏感性分析，不能把两条分支混成同一个 probe。

## 5. 可提取 representation

| 名称 | 分支 | 维度 | 用途 |
|---|---|---:|---|
| `online_projected` | online | 192 | 主要 global latent |
| `online_preprojector` | online | 192 | 判断 projector 是否丢失信息 |
| `online_global_pool` | online | 128 | 判断 encoder output linear/LN 是否丢失信息 |
| `online_roi_mean` | online spatial map | 128 | 主要 local/ROI representation |
| `ema_projected` | EMA target | 192 | 上一轮 current-state/readout 坐标敏感性 |
| `ema_preprojector` | EMA target | 192 | EMA projector 前敏感性 |
| `ema_global_pool` | EMA target | 128 | EMA global pooling 输出 |
| `ema_roi_mean` | EMA target spatial map | 128 | EMA local/ROI 敏感性 |
| `mask_geometry` | 输入 ROI mask | 9 | B2 几何 baseline |
| `raw_roi_intensity` | DCE7 + mask | 14 | B3 ROI 七通道均值/标准差 baseline |

ROI mean 使用 `adaptive_avg_pool3d(mask,[4,12,12])` 得到每个 spatial cell 的 ROI occupancy，再以 occupancy 对 feature map 加权平均；这比最近邻 resize 更能保存小病灶占比。空 mask 不产生伪造 feature。

## 6. 版本边界

本实验只加载上述 `best.pt`，不加载 `last.pt`，不修改 checkpoint。正式 test 患者只能由其唯一 outer-test fold 的同模型 checkpoint 产生 OOF prediction。所有 static/change target、StandardScaler、Ridge 和 baseline 均保持 patient-level train/validation/test 隔离；pCR、clinical、treatment 和 predicted future state均不进入 probe。
