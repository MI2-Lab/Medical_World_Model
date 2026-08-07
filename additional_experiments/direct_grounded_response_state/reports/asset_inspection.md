# Direct Grounded Response State 资产检查

## 1. 执行环境

- 分支：`feature/ispy-clean-corejepa`
- 检查时 HEAD：`629b9cdb6d9a713ca03cc7ff700c8d2fd71dc960`
- Python：3.11.14
- PyTorch：2.9.1+cu130
- CUDA runtime：13.0
- conda environment：`bowen`
- GPU：3 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，单卡约 96 GiB

检查时三张 GPU 均有其他用户的小型进程，故后续运行只使用剩余资源，不假定设备独占。工作区原有 `shortcut_audit/` 为未跟踪目录，本实验不读取、修改或纳入提交。

## 2. 锁定 cohort 与 split

五折文件为：

`${DGRS_DATA_ROOT}/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv`

SHA-256：`143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`。

文件含 4,040 行、808 名唯一 I-SPY2 患者和 5 folds。fold 0–2 的 train/val/test 为 525/121/162，fold 3–4 为 526/121/161；每折完整覆盖 808 人，每人恰好一次进入 test。375 名 FTV matched patients 在五折的 train/val/test 数分别为 247/59/69、239/69/67、240/52/83、242/61/72、225/66/84。

该文件是内部一致、已锁 SHA 的 seed-2026 candidate copy，但缺少 native clean split 的原始生成 provenance。后续结果只称为“锁定五折审计”，不称为 native numerical reproduction。

## 3. MRI cache 与 target

只读 cache 路径为：

`${DGRS_DATA_ROOT}/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96`

共覆盖 808 名 I-SPY2 与 156 名仅用于 base pretraining 的 I-SPY1。每个 NPZ 的 `x` 为 `[4,8,32,96,96]`；前 7 通道是 DCE，最后一通道是 binary ROI mask。808 人的 3,232 个 patient-visits 中有 41 个空 mask，涉及 33 人，因此 G2/G4 必须记录 empty-mask GAP fallback。

measurement mapping：

- `radiomics_patient_overlap.csv`：SHA-256 `91b575c9e7e351312b8181a091bdffd2d1f61b88b5a98ac3d78d54c94b63da6b`；
- `radiomics_transition_targets_raw.csv`：SHA-256 `26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d`；
- canonical raw-target mapping hash：`512749ccf986de4af4c0109b4ce060c61a90112816895a2ae7423784ea60de4e`。

375/808 人有四访完整 FTV/LD/sphericity/BPE。FTV 全部为正，四访合并范围约 0.012–471.314 cc。

## 4. G0 checkpoint 与 frozen feature

G0 复用 `radiomics_next_change/checkpoints/m0_final/fold_<k>/best.pt`。五折 checkpoint SHA-256 为：

| Fold | best epoch | checkpoint SHA-256 |
|---:|---:|---|
| 0 | 3 | `3d1ee55defd7dcd0306cf673aac64c39f6484518df77bd0abe6d4771cbad40d3` |
| 1 | 3 | `5c4ae970c3705d0b0d12b8f2133ef9c67649660cae2d5d3c5b14212e8e47d290` |
| 2 | 2 | `51a47dbae1321d24d0081c427eb67c469ccc5101b97c10e46604b595d44b78c0` |
| 3 | 2 | `d759bfbf68c704ce8d669e561900986349d1df15e94683d65ec35aa1dba25d15` |
| 4 | 3 | `64ba9c50ff81a50dd52b9f431cf0f48c2b2eaebc3c531f8dba3aa4763913eeb0` |

它们是 DCE8、base channels 16、latent 192、M0 Next-State+SIGReg 的 ROI-assisted checkpoint，source training commit 为 `c413ec86…`。复用时将同时记录 source commit 与本轮 execution commit。

对应 OSRA frozen feature 的 SHA-256 为：

| Fold | `observed_features.npz` SHA-256 |
|---:|---|
| 0 | `43aeb11a99720dd267fd81f9bfd22d25e1e72783ed5fe7154ef758bc0a9ac0bd` |
| 1 | `597e90d954892f5b5d8c44b25e27367c9eddd8e498d6e4b84839691fe6e3f6d0` |
| 2 | `ab00f824854eb98f08daab15234d070c22ffab4e4fec090530e7ba316b24e715` |
| 3 | `2daf3f3b45fae942adc879d0c87279bc5f30ee34ec2f856dbbbb3fdc770df4d5` |
| 4 | `af9dee4566ece0039d8f6cf8934797f000bdc5d2ef4ca4369ed8060959aa7f93` |

每折 feature 覆盖同一 808 人×4 visits；本轮 G0 primary representation 取 `online_preprojector [808,4,192]`。旧 G0 pCR 特征含 transition prediction，与本轮 observed-state-only contract 不同，不能复用旧 pCR 数值。

## 5. 已验证的 probe transform

Formal static probe 复用 OSRA 的 per-timepoint train-only transform：

- fold 0–4 SHA：`22b16983…e5e3`、`165a81ec…26c`、`b2cfdd8e…4a91`、`1621665e…9cb`、`e7e87975…1d278`。

Formal adjacent-change probe 复用 radiomics next-change 的 train-only transform：

- fold 0–4 SHA：`7b7ab138…e37`、`d3d89808…ed4`、`7c768b09…6009`、`681a2618…5ce`、`60f80445…837c`。

两组 transform 均只用 outer-fold training patients 拟合。新的 shared FTV training head 需要跨四访一致的单一尺度，故仍会在新目录内另建每折 pooled-four-visit `ftv_transform_fold_<k>.json`；该训练 transform 不替代 formal per-timepoint probe transform。

## 6. 关键风险与处理

1. DCE7 cache 仍是 ROI-centered crop，因此去掉 mask channel 不等于消除全部 ROI preprocessing prior。
2. Normalized ROI pooling 不显式暴露 mask volume，但 support 仍决定取样位置；最终只能声称“没有显式 geometry scalar 路径”。
3. 375 名 measurement matched patients 只占 46.41%，且已有 complete-case bias；所有 MRI patients 必须继续贡献 base loss。
4. G0 input contract 与 G1–G4 不同。正式 primary comparisons 是 G3−G1、G4−G2；G4−G0 只作描述。
5. 旧 M0/OSRA patient-level features、predictions 与 checkpoint 受 `.gitignore` 管理，只在授权环境存在；本轮所有引用都必须锁路径和 SHA。
