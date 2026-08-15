# Patch-token 治疗条件世界模型：最终报告

**最终分类：`POOLED_LOCAL_REMAINS_SUFFICIENT`。**

本实验完成了 2 个独立训练种子 × 5 个外层折的 A1 PATCH3 矩阵，并与已确认的 A0 LOCAL3 冻结基线做成对比较。世界模型训练、掩码、模型选择和 PCA 均未读取 pCR；pCR 只在十个检查点和无标签表征变换全部冻结后进入下游探针。治疗变量仅表示 assigned-regimen conditioning，不构成因果治疗效应。

## 结论摘要

- 种子 2026：实际时间余弦 0.3908，循环乱序 0.3901，差值 +0.0008；归一化 MSE 相对改善 +0.1%；目标/预测 token SD 为 0.9762/0.4826。
- 种子 3026：实际时间余弦 0.4092，循环乱序 0.4085，差值 +0.0008；归一化 MSE 相对改善 +0.1%；目标/预测 token SD 为 0.9783/0.4861。

## 十个问题的直接回答

1. **Patch-token JEPA 是否稳定训练？** 未达到预注册稳定性门槛；Gate A = FAIL。
2. **是否优于 LOCAL3 的静态 FTV？** 2026: 0.531→0.425 (Δ -0.106)；3026: 0.513→0.381 (Δ -0.132)。
3. **是否优于 LOCAL3 的ΔFTV？** 2026: 0.340→0.287 (Δ -0.053)；3026: 0.300→0.262 (Δ -0.038)。
4. **是否改善 MRI-only pCR？** 早期三前缀 AUROC 宏平均：2026: 0.526→0.538 (Δ +0.012)；3026: 0.533→0.516 (Δ -0.017)。
5. **是否增加超越临床 C的信息？** 2026 早期均值 ΔAUROC -0.025；3026 早期均值 ΔAUROC -0.031。
6. **是否增加超越临床+因果前缀 FTV的信息？** 2026 早期均值 ΔAUROC -0.098；3026 早期均值 ΔAUROC -0.111。
7. **未来 token 误差集中在哪里？** 按固定坐标带跨种子/访视均值，outer_local=1.2197，inner_local=1.1655，central=1.1053；最高为 `outer_local`。这些带不是病灶或瘤周区域。
8. **空间 token 是否找回 pooling 丢失的信息？** Gate C = FAIL；仅在同一端点、双种子方向一致且平均增益 ≥0.03 时才回答“是”。
9. **增益是 response-only 还是 phenotype-complementary？** 最终分类为 `POOLED_LOCAL_REMAINS_SUFFICIENT`；Gate D = FAIL。
10. **是否应替换 pooled LOCAL state？** 不应；按预注册规则保留 pooled LOCAL3。

## 门控

- Gate A `PATCH_DYNAMICS_VALID`: FAIL
- Gate B response preservation: FAIL
- Gate C `PATCH_STATE_ADDS_INFORMATION`: FAIL
- Gate D `PATCH_STATE_COMPLEMENTARITY_SUPPORTED`: FAIL

## 解释边界

- T3 始终是 late/pre-surgery；早期结论仅限 T0、T0–T1、T0–T2。
- 808 人均为 complete-4-visit 选择队列；FTV 分析的 375 人又是测量完整子集，不能把两个人群的绝对指标作增量比较。
- 500 个 token 来自固定 64-mm LOCAL 正重叠支持，边界权重用于精确均值；单 token 理论感受野约 42.3×42.3×94.0 mm，并非细粒度独立病理块。
- C1B-H 使用 T0 定位中心和 header-based 纵向策略；残余运动可表现为 token 预测误差。固定中心/内/外带仅为描述性坐标带。
- `delta_t=1` 是名义相邻访视间隔，不是实测扫描日差。assigned-treatment-conditioned longitudinal latent modeling 不等于因果治疗效应。
- 折内预测先合并为 OOF 后评分；bootstrap 在外层折内按患者重采样 2,000 次。折、访视和端点均不作为独立重复，训练种子才是独立重复。
