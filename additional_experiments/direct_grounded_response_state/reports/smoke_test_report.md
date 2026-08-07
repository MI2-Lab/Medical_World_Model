# G1–G4 真实 cache smoke test 报告

日期：2026-08-07
分支：`feature/ispy-clean-corejepa`
执行 commit：`629b9cdb6d9a713ca03cc7ff700c8d2fd71dc960`
conda 环境：`bowen`
训练实现 SHA-256：`fb308f8a3cfe735ca1ef2e17e66367b11d4e6edc424bd01725cad200b780e750`

## 1. 结论

G1、G2、G3、G4 均在 fold 0 的真实 DCE cache 上完成 1 个 train epoch、1 次 validation、checkpoint 保存与严格重载。四个模型的 loss、gradient 与 representation spread 均为有限值；G3/G4 的 FTV loss 对 encoder、response projection 和 Linear head 均产生非零梯度；没有发现 mask 被送入 G1/G3 或任何 backbone 的路径。全部 smoke 硬检查通过，可以进入 fold-0 lambda pilot。

这里的“通过”仅表示实现与数值链路可运行，不表示 grounding 已有效。Normalized ROI mean 仍通过 mask support 选择位置，DCE7 cache 也仍是 lesion-centered crop；因此 G2/G4 不能称为 geometry-free。

## 2. 真实 cache 训练结果

每个模型从 canonical fold 0 train 中确定性抽取 16 名患者、从 validation 中抽取 8 名患者；抽样保证至少有一名 FTV-paired 患者。batch size 为 32，四个模型各实际运行一个 train batch 和一个 validation batch。测试 split 未参与 smoke 训练或选择。

| 模型 | λ_FTV | Val state loss | Val FTV loss | Val representation std | Encoder grad norm | Raw FTV→encoder grad | Raw FTV→head grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | 0 | 1.707315 | 0 | 0.160468 | 8.879539 | 0 | 0 |
| G2 | 0 | 1.666136 | 0 | 0.432547 | 8.593340 | 0 | 0 |
| G3 | 0.05 | 1.705258 | 0.269930 | 0.160966 | 8.849039 | 11.888794 | 3.667052 |
| G4 | 0.05 | 1.664918 | 0.270389 | 0.434733 | 8.510085 | 15.017303 | 4.302356 |

G1/G2 的 `grounded_patients` 与 `valid_ftv_visits` 均为 0；G3/G4 的 smoke train batch 各有 7 名 paired patients、28 个有效 visits，同时保留 9 名无 FTV 患者的 base loss。G3/G4 的 EMA target modules 全部 `requires_grad=False`。

四个 finalized checkpoint 均以 `torch.load(..., weights_only=True)` 严格加载。其本地 SHA-256 为：

| 模型 | Checkpoint SHA-256 |
|---|---|
| G1 | `81f7c0747111626d19d4197621a5a7fcf03d09ab9dc1a3d66b825a7f8e6215c9` |
| G2 | `147c61e792ed0cec97b94f893ca267450e51260687bd38be04ad31c02664048c` |
| G3 | `2ef323e24867d698a317307003ddb37cd3ccb9008aad7ef90f42a5902a994800` |
| G4 | `15cbb5e4c1ac664d7532d1575e7b46cd7a9ef6571985abfc68f162b77112f88d` |

Checkpoint 与 patient-level history 受 `.gitignore` 管理，不提交 GitHub；上表哈希用于授权环境内复核。

## 3. DCE7 与 mask 路由

真实 cache 样本在 loader 后满足：

- DCE tensor：`[4,7,32,96,96]`；
- 分离 ROI mask：`[4,1,32,96,96]`；
- 四个模型第一层均为 `Conv3d(in_channels=7, ...)`；
- G1/G3 传入 mask 会立即抛出 `ValueError`；
- G2/G4 缺少分离 mask 会立即抛出 `ValueError`；
- encoder 的 forward 签名只有 `image`，mask 在 spatial map 形成后才进入独立 pooling 函数；
- model forward 没有 FTV target 参数，FTV 不是推理输入。

固定 DCE 输入重复执行 encoder 时 spatial feature map bitwise 相同。State dict 中未出现 clinical、treatment、mask geometry、voxel count、explicit volume 或 FTV target 路径。

## 4. Normalized ROI pooling 不变量

独立张量检查结果：

| 检查 | 最大绝对误差 | 结论 |
|---|---:|---|
| `pool(F,M)` vs `pool(F,3.7M)` | `5.96e-08` | 通过 |
| 常量 spatial map、不同非空 support | `0` | 通过 |
| all-ones mask vs GAP | `0` | 通过 |
| empty mask fallback vs GAP | `0` | 通过且 finite |

这些检查排除了通过未归一化 feature sum 直接传递 mask volume 的路径。它们不排除 mask support 的位置、形状和 crop prior 对所取 image feature 的影响。

## 5. FTV 对齐、transform 与泄漏

- Fold 0 train/validation/test patient sets 两两无交集，每名 primary patient 的五折 test coverage 仍为一次。
- 五个 pooled-four-visit FTV transform 均已仅由各 fold outer-train patients 拟合；paired train patient 数为 `247/239/240/242/225`，有效 visit 数为 `988/956/960/968/900`。
- G1/G3 与 G2/G4 的初始化哈希分别完全一致；实际四个 smoke 的公共模块初始化哈希均为 `59415d1b16d26a38ed02a58a28bbedff20b9544d5f5bc7c8fe970941726518df`。
- G3/G4 的 target 只进入 loss；移除 FTV head 不改变 `encode_online` 的计算图或 `r/z`。
- Smoke checkpoint selection 只读取 train/validation state、FTV loss与 representation spread，没有读取 pCR 或 test feature/target。

五折训练 transform SHA-256 依次为：

1. `8df48a908a5d56f76a2dd1a5f52b7189b03ce64e60743f856ef14afca07ebd5b`
2. `6b582c2bb22e8208bc2e149eec032d179182fde212b94bcf6161bd274b38b4d4`
3. `fcdf72ea26da1ff49efbdc937c78761e41d54640dae20289ac73a193e9cee23a`
4. `a666b556e87c955214869547c6d54f083b8f975838c12461cc1158332532792c`
5. `cb207a387900cc9ebc3deb7dca8e448bdbea083aae495af07fd11200008d6a9c`

## 6. 复现命令

真实 smoke 使用 `scripts/train.py`，分别执行 G1/G2 后再把对应 baseline checkpoint 传给 G3/G4；最终硬契约由以下命令统一复核：

```bash
conda run -n bowen python scripts/run_smoke_checks.py
```

机器可读完整输出保存在本地 `metrics/smoke/smoke_checks.json`，其中 `status=passed`。该文件包含本地 checkpoint 引用且不提交公开 Git。
