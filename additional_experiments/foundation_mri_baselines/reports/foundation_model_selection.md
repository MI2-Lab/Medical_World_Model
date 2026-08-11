# Foundation model selection（正式 test 前冻结）

冻结日期：2026-08-11。冻结时尚未运行任何正式 pCR/HR/HER2/FTV outer-test probe，也未查看 foundation test 指标。

## 正式候选

| Model | Pretraining data/domain | 2D/3D | Native input | Checkpoint source | License | Parameters | Native resolution | 冻结理由 |
|---|---|---:|---|---|---|---:|---|---|
| MedicalNet 3DSeg-8 ResNet-50 | 8 个可枚举 3-D segmentation 数据集，1,638 volumes；MRI+CT，不含 I-SPY/pCR | 3D | 1 channel | [Tencent official repository](https://github.com/Tencent/MedicalNet/tree/2d880ef30f86d573b21adbaa7661161701382c2d) / [official GDrive archive](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view) | [MIT](https://raw.githubusercontent.com/Tencent/MedicalNet/2d880ef30f86d573b21adbaa7661161701382c2d/LICENSE) | 46,155,072 | Fully convolutional；official demo 56×448×448 | 唯一正式 3-D medical reference；语料可审计；共享单通道 encoder 保留全部 DCE7 |
| DINO v1 ViT-B/16 | ImageNet-1K，无标签 self-distillation；固定自然图像语料，不含 I-SPY | 2D | RGB 224×224 | [Meta official repository](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a) / [native checkpoint](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) | [Apache-2.0](https://raw.githubusercontent.com/facebookresearch/dino/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE) | 85,798,656 | 224×224, patch 16 | 强通用 SSL reference；公开、non-gated、训练语料边界比后续 web-scale DINO 更可审计 |

MedicalNet 3DSeg-8 的组成来自 [Med3D/MedicalNet 论文](https://arxiv.org/abs/1904.00625)：Brain、Hippocampus、Prostate、Heart、Liver、Pancreas、Vessel、Spleen，监督为 segmentation mask，不含 pCR。官方 archive 没有上游 checksum，且最初单文件链接已失效，因此 artifact immutability 是“取得时本地锁定”的 conditional pass：只接受本次 official archive 内的 `resnet_50.pth`，立即记录 SHA-256 并要求 318/318 strict load。若制度要求发布者预先提供不可变 digest，则该候选不满足制度 gate；本实验不以其他分数更高的模型静默替代。

## Checkpoint lock

| Artifact | Size (bytes) | SHA-256 | 正式使用 |
|---|---:|---|---|
| `MedicalNet_pytorch_files.zip` | 2,796,095,393 | `4ba2ece5f32a13b166e431b78e99052c9142f879f60a150baa54ba5068eaf84b` | 否，仅 acquisition archive |
| `resnet_50.pth` | 184,885,051 | `5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84` | 是，3DSeg-8 |
| `dino_vitbase16_pretrain.pth` | 343,242,485 | `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e` | 是 |

MedicalNet 的 `resnet_50_23dataset.pth` 明确不使用：官方没有公开 23 个组成数据集，不能严格排除 I-SPY。DINO 使用 Meta 原生 checkpoint/LayerNorm `eps=1e-6`，不使用 HF conversion 的 `eps=1e-12` 或未训练 pooler。

## 预注册 channel/spatial adaptation

- MedicalNet：七个 DCE channel 逐通道共享 frozen encoder；同一 native-grid layer4 map 分别 GLOBAL mean 与固定中心 64-mm overlap pooling；按 DCE7 顺序拼接成 14,336 维。
- DINO：固定三通道 `early, late, late-minus-pre`；32 axial slices；GLOBAL 完整 FOV 对称 pad，LOCAL 为固定中心 64-mm cube；均显式 resize 224。固定值域映射后取 `CLS + mean patches`，再 slice mean，得到 1,536 维。
- 两者均不访问 lesion mask、FTV、clinical 或 outcome；不得依据 test pCR 改 channel、slice、pooling 或 checkpoint。

## 冻结前拒绝候选

| Candidate | 拒绝理由（与 test 结果无关） |
|---|---|
| BiomedCLIP | PMC-15M 来自全量 PMC-OA figure-caption pairs；未发布 I-SPY/pCR 排除清单。I-SPY 论文图像或 outcome-bearing caption 的污染风险无法消除，即使 checkpoint 为 public/MIT 也不满足本任务的污染 gate。 |
| DINOv2 | LVD-142M 从大规模 crawled web images 检索建立，精确训练图像清单不公开；无法做试验级 I-SPY 排除。 |
| DINOv3 | LVD-1689M web/Instagram 语料不可枚举；HF manual-gated；采用 Meta custom license，匿名获取和标准许可均不如 DINO v1。 |
| MedicalNet 23-dataset | 官方只说明“23 datasets”，未列公开数据清单；无法严格证明没有 I-SPY。 |

BiomedCLIP/DINOv3 权重曾在候选核验阶段下载或检查结构，但没有接触 patient tensor、没有产生正式 representation，也不会进入任何结果表。

