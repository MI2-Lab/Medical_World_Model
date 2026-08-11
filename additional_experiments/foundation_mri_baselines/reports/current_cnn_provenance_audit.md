# Current CNN comparator provenance audit

审计日期：2026-08-11。该审计在新 pCR test evaluation 前完成，只读核验现有 `local_global_response_state_pilot` 资产，没有修改旧实验。

## 正式 comparator

Primary current-CNN comparator 固定为 seed 2026 的 `GAP0`（GLOBAL）和 `LOCAL0`（LOCAL）：两者均为 `grounded=false`，训练/选择不使用 pCR、FTV 或 test data，selected epoch 的 FTV loss 为 0。`GAP3/LOCAL3` 是直接受 FTV loss 塑形的 grounded representation，因此不能作为与无 I-SPY outcome/FTV 预训练的 foundation encoder 的 image-only primary comparator。

10/10 个 fold asset 均为 float32 `[808,4,192]`，全 finite、808 unique；每 fold 的 embedded split 与正式 manifest 逐患者一致。五折 test 集合合并后为 808 unique patients，每人恰好一次 test。不同 fold 的原始行序不同，正式 loader 必须按 patient ID 重排，禁止按 row index 拼接。

| Fold | Train | Validation | Test |
|---:|---:|---:|---:|
| 0 | 525 | 121 | 162 |
| 1 | 525 | 121 | 162 |
| 2 | 525 | 121 | 162 |
| 3 | 526 | 121 | 161 |
| 4 | 526 | 121 | 161 |

## Asset hash table

| Arm/fold | Epoch | Feature SHA-256 | Checkpoint SHA-256 | Selection SHA-256 |
|---|---:|---|---|---|
| GAP0/0 | 3 | `affcba495e1935d5805f10732e982f65a2f5f62c3a29b7874c8d28e9825faaf3` | `193ad8230c289f762b76a46b62ad6824818095c3966b1d2f208dbaa8e6b40b4f` | `6882917356655965821244b91f1a3d0806e43cfcb022bef49b5dd3ad1bac03f7` |
| GAP0/1 | 3 | `a8b0457a72ffb032c803c16b3ebd1ee71b2193cc78bca6b22e709dcb912c6719` | `74b7443f918bb29ee59b170c725df38d1f9ffebaf086a6677dc7d9a385a97d27` | `ece4e5023f93f546bd9fb98927770a21be560eeaccfe68b2b4663a6455e1f13a` |
| GAP0/2 | 3 | `fbda603a5ac9ff1d51149972d6528900a81a4acae6c812bc930507758816e789` | `7e138cb2d3e88a536e240a5f5da3fe8bb0bdf016c8053211dd6d715032ae63e7` | `a867031daeb6a36cc6d7bf42013b1fb15702e17b75dbe393242c546371e9396e` |
| GAP0/3 | 3 | `af2157e79e6549427ef8093fb7f31771801924de2a1dc400f80692111e130cbb` | `6f419376866756eb2a41c7a1541cde67bd826416d0c6ba010c74cae14f417b32` | `4001d2bfcf5781920f159796bc122fbcb7dde6e0af32e7bba06dfaebaa8a4c94` |
| GAP0/4 | 3 | `4599eed75a60ff277d715b7c97043b0dcc9eff605a68c3724f70e4141f4a6fe9` | `2bd040f40ea872177e709e2987d41b00a87c2eaebab333ebc965b8c133df2b3b` | `81d7a98595b3b4de82eda4bf91d2328ee553e0805dee4b81e83e69d71167ea6c` |
| LOCAL0/0 | 3 | `360cd7b4e642b2a44110cb42d246c4c2b36c8364fdb007c3a30221de2de5b143` | `2c16fc0311c075f464352ad69aeeef2f6ad2d086168ea618f6fc95f14e1dcf3c` | `cbc6f0e6c62c47f5364d148daafca1a4632dcbc505ddccbde6fcaa7013c10187` |
| LOCAL0/1 | 2 | `525db16f5e7071713289a5cec4fc8c8266c2855159886c886904de1348f55a53` | `ec315e2d40b72b97cc9813a4fabbd7adff2495189e60c34ccdf2927a2873b386` | `c816c7b2e0a35057af8778bbe6af2d9f42c3633df8466ac7b05e1a4ffed706f2` |
| LOCAL0/2 | 2 | `6c1d6c02da00c0e1600e15bf09eed1050146805139bbd87640b7ad9588b93aeb` | `f9fa38d112f9fa4bbfa39281a399a795a358c117d0e56f57387c1dc3a508d65e` | `96b322e3adfe165029a7478bdccceb3fab8817015886ebe3f9a623d672166cf3` |
| LOCAL0/3 | 2 | `f6f0210e6dfafffd94fe88a180e2d4618a97b40ea499fb6670c6ddaf6d59f99c` | `73a824ef99a5bd579ab3f0460ed312602cfa97e237cc3bf774261b07bafed17c` | `b54ec7e10e320fec54cb317da284fe1298be5bde3a80a0c43ea509b610a38588` |
| LOCAL0/4 | 2 | `a8ad868a5972e287d0922a53c6c4dd99fb13a9e4b5aa7cfed9e1e695fff95cef` | `f2c34863ab7099c9e6090e266ba8083653de291e5f4abb9e0a45858f1a7c9022` | `a973d57519a0540c1670d6bbfba18414ab5e3e0a463a312356dc36a6e189a8e2` |

## Common locked provenance

- Feature implementation：`8b27b0c452514756a7b6af91b25a26534ab62c50070f92e1498e7a94dd314e87`
- Current data provenance：`2fbaa0b40eb988aded4b35c0fcc03de2eceff68eeeddfdb3d2d04306ec80ed07`
- Preregistration lock：`d2fcbd6e92300debe462da1968d74f4809a03bfdccaea2cd82edfead846c4daa`
- Stage-A sentinel：`0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb`
- Fold manifest：`143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`
- C1B cache manifest：`672ad7436b19f30a89640a2b36504f1e7fbaaff83fd07bc058c008b204d2a3c9`

正式 loader 对每份 asset 强制 adjacent 27-key metadata schema、feature/checkpoint/selection/lock digest、arm/fold/seed/spatial identity、`ftv_head_called=false`、`test_labels_used=false`、selection 中 `test_data_used=false`、`pcr_used=false`、`delta_ftv_used=false`，以及 embedded split 与正式五折逐患者一致；任一 drift 均 fail closed。

审计时仓库内不存在可直接引用的 matched GAP0/LOCAL0 pCR 或 +clinical 正式结果。本实验因此按统一 probe 首次计算它们，而不是复用不同协议或不同人群的数字。
