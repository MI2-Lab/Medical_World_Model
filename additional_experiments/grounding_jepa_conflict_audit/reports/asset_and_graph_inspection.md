# G1/G3 资产与 loss graph 核验

## 环境与仓库起点

- 分支：`feature/ispy-clean-corejepa`
- 审计起点提交：`703a2d6febec93f75298183e3d170f18ad666589`
- Python：3.11.14
- PyTorch：2.9.1+cu130
- CUDA runtime：13.0
- GPU：3× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（每张 97,887 MiB）
- conda 环境：`bowen`
- 起始工作树中仅有既存未跟踪 `shortcut_audit/`；本实验不修改它。

## 既有正式资产

`g3_multiseed_generalization` 本地授权环境中完整保存：

- 25 个 G3 selected `best.pt`，覆盖 5 training seeds×5 folds；
- 对应 25 个 G1 selected checkpoint；
- 50 个 `last.pt`、25 个 G3 `fallback.pt`；
- 50 个正式 history 与 selection，其中本审计只读取 25 个 G3 history；
- 五个 fold-specific FTV transform；
- 已验收的 stability、seed-fold probe 与 downstream 聚合表。

25 个 G3 selected checkpoint 均为 finalized、G3、DCE7、GAP、`lambda_FTV=0.25`、batch size 32，且同 fold 的五个 seed 具有完全一致的 patient split。Smoke history 不进入正式审计。

## 真实 forward 与 loss

`model.py` 确认 observed representation 是 online pre-projector `r`：

```text
DCE7 -> 4-stage 3-D residual encoder -> GAP
     -> Linear(128,192) + LayerNorm -> r
```

Branch A：`r -> VisitProjector -> causal transition -> predicted_next`，与 detached EMA target 比较得到 normalized next-state loss；online projected state 同时进入 SIGReg。因此训练中的 raw base component 为：

```text
L_base = state_loss + 0.09 * SIGReg
```

Branch B：`r -> Linear(192,1)`，在有 FTV 的患者内先对有效访视平均 SmoothL1，再对患者平均，得到 raw `L_FTV`。总目标严格为：

```text
L_total = L_base + 0.25 * L_FTV
```

Validation base safety gate 使用不含 SIGReg 的 `state_loss`；本 conflict audit 的 `g_base` 使用真实训练目标中的完整 `L_base`，并在报告中保留这一差异。

## 参数接收关系

| 参数组 | 参数张量数 | 参数量 | 接收 base 梯度 | 接收 FTV 梯度 | shared |
|---|---:|---:|---|---|---|
| `encoder.features.0` | 7 | 10,112 | 是 | 是 | 是 |
| `encoder.features.1` | 7 | 42,112 | 是 | 是 | 是 |
| `encoder.features.2` | 7 | 168,192 | 是 | 是 | 是 |
| `encoder.features.3` | 7 | 672,256 | 是 | 是 | 是 |
| `encoder` overall | 28 | 892,672 | 是 | 是 | 是 |
| `response_projection` | 4 | 25,152 | 是 | 是 | 是 |
| `projector` | 6 | 148,800 | 是 | 否 | 否 |
| `transition` | 43 | 1,187,904 | 是 | 否 | 否 |
| `ftv_head` | 2 | 193 | 否 | 是 | 否 |
| `target_encoder` | 28 | 892,672 | 否，冻结 | 否，冻结 | 否 |
| `target_response_projection` | 4 | 25,152 | 否，冻结 | 否，冻结 | 否 |
| `target_projector` | 6 | 148,800 | 否，冻结 | 否，冻结 | 否 |
| `all_shared` | 32 | 917,824 | 是 | 是 | 是 |

FTV head 直接读取 pre-projector `r`，所以 projector 与 transition 不在交集；target 分支在 `no_grad` 中运行并 detach，也不在交集。

## Checkpoint trajectory 能力边界

每个 G3 run 有 selected、fallback、last 文件。当前 25 格中 fallback 与 selected 都对应同一 epoch，并具有相同 model state，只是序列化 metadata 不同，因此不能当第三个时间点。Last 是 selected 后继续 patience 训练得到的不同状态。

可做：selected 与 last 的两点 post-hoc gradient comparison，以及完整 CSV history 的 loss/exposure trajectory。

不可做：early/每 epoch 权重 gradient trajectory、bit-exact resume 或原 minibatch replay。Checkpoint 未保存逐 epoch model state、DataLoader generator state、dropout mask或 SIGReg 随机 direction state。最终报告必须明确这一限制，禁止用重新训练补齐。

## 隐私边界

Checkpoint metadata、fold manifest、raw FTV 与私有 batch membership 含患者标识或本机路径，均不得提交。公开 batch manifest 只允许整批 HMAC 与计数；所有正式梯度表必须不含 patient ID、pCR、clinical、treatment 或 subtype。

