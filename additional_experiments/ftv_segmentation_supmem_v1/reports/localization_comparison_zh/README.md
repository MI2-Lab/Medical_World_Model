# I-SPY2 FTV 定位与裁剪对比报告

## 一句话结论

冻结记忆（frozen-memory）SAM2 在完整 I-SPY2 队列上已经是明显优于旧自动增强 ROI 的病灶定位器，但**还不能把 released mask 从生产裁剪流程中直接移除**。原因不仅是模型定位误差：即使把 released lesion 的当前随访质心作为 oracle 中心，固定 `32×96×96 (z,y,x)` 原生体素窗口仍有 30.8%–40.6% 的非空真值体积不能达到 0.9 containment。这说明裁剪窗口本身也是主要瓶颈。

本文件把两个互补问题整理在一起：

1. `mask_comparison/`：模型或旧启发式生成的区域与 released FTV mask 是否重叠（Dice）。
2. `crop_containment/`：在固定训练窗口内，真实病灶有多少被保留（containment）。这是生产裁剪真正关心的问题。

所有数字和图像均来自同一版已完成的 `frozen_memory` 五折实验；没有使用仍在调参的 `tuned_a/b/c` checkpoint。

## 数据、可复现性与病例选择

- 队列：808 位 I-SPY2 患者、4 个随访（T0/T1/T2/T3），共 3,232 个体积。
- 真值：released `ftv_mask_nifti`；显示背景为缓存的 early−pre DCE enhancement。
- 模型预测：每位患者严格只使用其在 CV CSV 中唯一 `split=test` fold 的 `runs/frozen_memory/fold_k/best.pt`。该映射在生成报告时逐患者断言。
- 可视化的六位患者不是人工挑选：在 pCR=0 与 pCR=1 内分别按 T0–T2 平均 held-out Dice 的 90%、50%、10% 分位选取最近者，且四次随访均要求真值非空。pCR 只用于选例及图中标签，从不输入任何模型、阈值或指标。
- T2 有 17 个、T3 有 47 个 released truth 为空；这些体积的 containment 未定义，保留在 JSON 中并从相应汇总中排除。四次随访均非空的子队列为 753 人，应视为较容易的子队列。

## A. Mask / 区域定位比较

### 比较的三个区域

| 方法 | 区域如何获得 | 是否读取 released mask | 作用 |
|---|---|---:|---|
| Released I-SPY2 mask | 发布的 FTV mask | 是 | 真值 |
| Supervised frozen-memory SAM2 | 整个原生体积上的 held-out 预测 logits，重采样到原生网格并以 `logits > 0` 阈值化 | 否 | 学习式定位/分割 |
| Automatic enhancement ROI | 原始 4-D DCE 上的高增强启发式、3-D 连通域筛选及膨胀 | 否 | 旧的 mask-free stub |

### 结果

完整 808 人队列中，supervised frozen-memory 的平均 Dice 为 **0.607 / 0.526 / 0.309 / 0.216**（T0/T1/T2/T3）。这显示基线前和早期治疗随访的病灶定位具有实用信号，但后续随访明显变难。

旧 automatic ROI 的独立 40 人基线在每次随访的 Dice 中位数均为 **0.0000**，全体积质心距离中位数为 **97 mm**。六个可视化病例中它仅在 24 个展示切片中的 7 个出现；缺失切片在图里被明确标注，而不是静默省略。

`mask_comparison/comparison_grid.png` 显示六位患者 × 四次随访的全原生 FOV。绿色填充/轮廓为 released truth，蓝色实线为 supervised prediction，橙色虚线为 automatic ROI。`selection_context.png` 证明六例覆盖两种 pCR 状态下的高、中、低质量范围；六张 `case_*.png` 提供更大的逐患者视图和体积指标。

### 解读

Dice 的进步并不自动等于适合生产裁剪：一个预测可以具有较低 Dice，但质心仍足以把病灶放进训练窗口；反过来，一个平均 Dice 较高的预测也可能在 z 或边缘上漏掉足以破坏裁剪的部分。所以下一节把定位问题改用 containment 重新评估。

## B. Crop placement / 裁剪放置比较

所有方法放置同一个 `32×96×96 (z,y,x)` 原生体素窗口。中心会四舍五入到体素索引，x/y 各从中心向两侧取 48 个体素，z 各取 16 个体素；越出图像边界的区域由零填充。物理毫米范围由每位患者的 spacing 决定，不能把相同像素框误认为相同的物理窗口。

Containment 的定义完全一致：

`containment = crop 内 released 真值体素数 / 全部 released 真值体素数`

### 五种中心策略

| Arm | 中心的具体来源 | 是否读 mask | 随访间处理 |
|---|---|---:|---|
| `oracle` | 当前随访 released FTV mask 中所有体素 `(x,y,z)` 坐标的均值 | 是 | 每次随访独立计算；理论上限 |
| `pipeline` | T0 released bbox 的包围盒中点 | 是 | 按原图尺寸比例投影到 T1/T2/T3；当前生产方案 |
| `learned_pervisit` | 当前随访 held-out supervised predicted mask 的 `pred_centroid_xyz` | 否 | 每次随访独立更新 |
| `learned_t0anchor` | T0 held-out predicted mask 的质心 | 否 | 使用与 pipeline **完全相同**的尺寸比例投影 |
| `auto_stub` | 当前随访原始 4-D DCE 的 automatic enhancement ROI 质心 | 否 | 每次随访独立计算 |

投影采用上游 `_project_center`：每个轴的 T0 归一化坐标乘以目标随访该轴长度，再裁剪到图像内。`pipeline` 与 `learned_t0anchor` 唯一的中心来源差异，正是 released T0 bbox 与模型 T0 预测质心的差异；因此它们是最直接的 mask-dependence 对照。

`auto_stub` 在每次随访的原始 4-D DCE 中，以第一个时相为 pre，寻找高相对增强候选体素，进行 3-D 连通域筛选、选择最高分组件并膨胀。为避免给调参作业带来负担，auto stub 使用固定随机 **n=40**（seed 2026）的队列；六个展示病例必要时额外计算，仅用于画框，不进入该 arm 的队列汇总。

### 完整非空真值队列：中位 containment 与失败率

失败率是 containment `<0.9` / `<0.5` 的体积比例；n 对 oracle、pipeline 与两种 learned arm 分别为 T0/T1/T2/T3 的 808/806/791/761，auto stub 为固定 n=40。

| Visit | Oracle median / <0.9 | Pipeline median / <0.9 | Learned per-visit median / <0.9 | Learned T0-anchor median / <0.9 | Auto stub median / <0.9 |
|---|---|---|---|---|---|
| T0 | 0.973 / 30.8% | 0.977 / 33.9% | 0.928 / 44.3% | 0.928 / 44.3% | 0.000 / 100.0% |
| T1 | 0.979 / 29.7% | 0.879 / 52.5% | 0.929 / 45.3% | 0.830 / 59.9% | 0.000 / 97.5% |
| T2 | 0.966 / 36.8% | 0.807 / 59.3% | 0.750 / 62.2% | 0.742 / 64.1% | 0.000 / 100.0% |
| T3 | 0.956 / 40.6% | 0.750 / 64.8% | 0.638 / 70.8% | 0.665 / 71.5% | 0.000 / 100.0% |

完整 IQR、`<0.5` 失败率以及 all-four-non-empty 子队列表格见 `crop_containment/README_original.md`；原始逐体积记录见 `metrics/crop_containment.json`。

### 结果分析

1. **T0：模型尚未达到现有 pipeline 的裁剪可靠性。** Learned per-visit 的 `<0.9` 失败率为 44.3%，比 pipeline 的 33.9% 高 10.4 个百分点；而 T0 anchor 与 per-visit 相同，因为两者在 T0 使用同一个模型预测中心。
2. **T1：逐次随访模型定位优于 released T0 bbox 的固定投影，但仍不足以移除 mask。** Learned per-visit 的失败率 45.3%，低于 pipeline 的 52.5%；但仍接近一半体积达不到 0.9 containment。T0 anchor 的 59.9% 进一步说明，治疗开始后不能只依赖 T0 的模型位置。
3. **T2/T3：所有非 oracle 的局部策略都出现明显退化。** T2/T3 的 learned per-visit 中位数降到 0.750/0.638；T0 anchor 为 0.742/0.665。病灶缩小、形态改变、增强减弱及不同随访成像差异共同放大了定位误差。
4. **旧 auto stub 不能作为替代。** 它在 n=40 子集中几乎从不包含病灶；这与 97 mm 中位质心距离及 mask 比较中的零 Dice 一致。
5. **窗口合同本身需要修订。** 即使 oracle 的中心正确，T0–T3 仍有 30.8%–40.6% 的体积 `<0.9` containment。对于 learned-pervisit 的 `<0.9` 失败，约 80%–85% 同时跨越多个轴；单独 z 逃逸约 10.6%–15.6%，高于单独 x/y，但数据不支持“只有 z”这一解释。后续优化应同时评估更大的或各向异性的 z 范围、边缘容忍度，以及按不确定性自适应的窗口大小。

### 图像阅读方式

`crop_containment/crop_grid.png` 与 mask comparison 使用同一组六位患者。每个单元格包含：

- 上：全 FOV 的 axial x–y MIP；下：coronal x–z MIP。后者不可省略，因为 z 方向的窗口厚度只有 32 层。
- 绿色：released lesion；洋红：落在 learned-pervisit crop **外**的真值病灶。
- 灰色点线：oracle；橙色实线：pipeline；蓝色实线：learned per-visit；紫色虚线：auto stub。
- 面板中的 `O/P/L/A` 分别表示上述四种可视化策略的 containment；每个单元格还列出该患者实际物理 FOV（z/y/x mm）。

## 建议

1. 近期不要以 auto stub 替换 released ROI，也不要仅用当前 learned centroid 无条件替换 pipeline crop。
2. 将 learned per-visit 作为 T1 的候选辅助定位，而非单点替换；应依据 containment 风险或预测不确定性触发回退/扩大窗口。
3. 优先做 crop-contract 消融：扩大 z、允许不对称边界、或按 spacing 与 lesion-scale 设置物理毫米窗口。oracle 的失败率表明这一步比单纯改善 localizer 更基础。
4. 用这套同一 held-out-fold 审计在修改 crop contract 后重新报告全队列 median、IQR、`<0.9`/`<0.5` 尾部失败率，而不能只报告 Dice。

## 文件清单

```text
localization_comparison_zh/
├── README.md                         # 本中文综合报告
├── mask_comparison/                  # Dice / 区域定位图与原始报告
├── crop_containment/                 # containment、逃逸轴图与原始报告
└── metrics/                          # 两套机器可读的逐体积指标
```

`README_original.md` 保留原始英文方法和数值叙述；本报告是其中文整合与结果解读。
