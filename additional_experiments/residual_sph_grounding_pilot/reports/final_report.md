# FTV + residual-SPH grounding pilot 最终交付报告

## 当前结论

本次交付已完成科学预注册、实现、泄漏防护和可执行流水线，但**尚未产生正式实验结果**。截至 2026-08-12 14:25 EDT，三张 GPU 仅剩约 13.5、3.2 和 9.8 GiB 显存，均低于预注册的 60,000 MiB 安全阈值；同类 LOCAL 3-D 训练通常需要约 60 GiB 以上。因此资源保护器明确拒绝启动 30 个新训练单元，也没有中止、挤占或修改任何已有任务。

这不是阴性结果。Gate A–D 均为 `NOT_EVALUATED`，最终科学分类为 `NOT_ASSIGNED`。在完整 S1/S2 表征冻结之前，pCR 防火墙保持关闭。

## 先验证据与边界

- 五种子研究明确给出 `LOCAL_MULTISEED_CONFIRMED`；LOCAL3 是当前有依据的 grounded LOCAL 参考。
- LOCAL3 相对 LOCAL0 的静态 FTV macro Spearman 平均增益为 +0.0297，观测 ΔFTV 平均增益为 +0.0322，优化安全为 25/25；这是小而稳定的 FTV grounding 增益。
- LD 有描述性下游相关性，但没有稳定的 FTV-independent 图像可解码性。BPE 因 `FOV_OBSERVABILITY_UNVERIFIED` 继续阻断。
- 既往 residual-SPH 通过的是 T0 的 `LOCAL3/Z4`（256 维全 LOCAL 空间均值+标准差），不是当前 192 维 `Z2` response state 的稳定通过。因此本试验是风险明确的新探索，而不是既往结论的直接确认。
- 动态 SPH 证据失败；实现中不存在 ΔSPH、百分比变化、future-SPH 或跨访视 SPH 监督。
- residual target 严格采用每个 outer fold、每个访视的 train-only 1/99 winsor、SPH identity/population-z、FTV log1p/population-z、Ridge(alpha=1)、epsilon 再 population-z。没有在自然单位中直接做 `SPH-f(FTV)`。
- 经核查，confirmed LOCAL transition 是 image-only。实现保留该事实，没有加入 treatment/clinical 输入。

## 12 个要求问题的当前回答

1. **LOCAL response 性能是否保留？** 尚不能判断；Gate A 未评估。
2. **raw-SPH grounding 是否有效？** 尚不能判断；S1 的 10 个单元尚未运行。
3. **residual-SPH grounding 是否有效？** 尚不能判断；S2 的 10 个 primary 单元尚未运行。
4. **residual grounding 是否优于 raw grounding？** 尚不能判断；E4/Gate C 未评估。
5. **S2 是否改善 FTV-independent morphology 表征？** 尚不能判断；T0 SPH_res Spearman 的 E3 未产生。
6. **是否影响静态 FTV？** 尚不能判断；E1 未产生。
7. **是否影响观测 ΔFTV？** 尚不能判断；E2 未产生。ΔFTV 仅用于冻结后的评估。
8. **是否改善 MRI-only pCR？** 未评估；表征尚未冻结，pCR 读取被禁止。
9. **是否增加 clinical 之外的信息？** 未评估。
10. **是否增加 clinical+FTV 之外的信息？** 未评估；Gate D 未评估。
11. **SPH 是否值得保留为 auxiliary target？** 当前证据不足，不能决定。
12. **是否值得做五种子确认？** 当前不能授权；只有 A+B+C 通过后才可授权。

## 已完成的审计与实现

- S0/S1/S2/S2-L10 严格矩阵：两种子 × 五折；primary `lambda_sph=0.05`，唯一 sensitivity 为 0.10。
- 科学锁与 42 个 implementation/test Python 文件的实现锁均通过哈希校验；完整测试为 70 passed。正常运行入口在任何正式单元开始前都会重新验证两层锁。
- S0 构建器与 confirmed LOCAL3 的 state dict 一致；两种子的共享初始化哈希与确认研究完全一致。S1/S2 仅新增 `Linear(192,1)`，即 193 个参数。
- 10/10 confirmed LOCAL3 S0 checkpoint、selection、FTV transform、feature 与 metadata 已完成逐文件哈希和来源 ancestry 审计，source classification 为 `LOCAL_MULTISEED_CONFIRMED`。
- 物理 B4 × accumulation 8 的 logical B32 训练器分别核算 FTV 与 SPH 的有效患者分母，每个 logical batch 仅进行一次 clip、AdamW step 和 EMA update。
- checkpoint selector 对 SPH、test、ΔSPH 和 pCR 不敏感：先满足 `val_state <= 1.05 × paired S0`，再以 validation FTV loss 选择。
- 375 人 residualizer 只读取 allowlist FTV/SPH 字段；训练阶段没有读取 pCR、HR、HER2、clinical 或 treatment。
- 5/5 outer-fold、20/20 fold×visit residualizer 已按 train-only 规则生成并通过既往 Goal-6 参数 parity 测试；没有保存 patient-level residual target。
- S2/seed-2026/fold-0 的只读 preflight 已通过：broader JEPA train=664、primary train=525、SPH train=247、validation=121、SPH validation=59；没有打开 image cache，也没有开始训练。
- frozen-state probe 先在每个种子内拼接五个 outer-test fold，再计算指标；种子是独立重复单位。
- pCR 程序只有在所有 checkpoint、feature、probe 和 residualizer 被 `representation_freeze.json` 哈希冻结后才可读取 clinical/pCR 表。

## 尚空缺的正式结果表

以下表必须由冻结流水线生成，当前不得用先验结果或占位数字填充：

- target residualization audit：已生成 [residualizer inventory](../manifests/residualizer_inventory.json) 与 [20 个 aggregate fits](../metrics/residualizer_fits.csv)；
- optimization safety：0/10 primary S2 单元可评估；
- static FTV 与 observed ΔFTV：无 S1/S2 OOF 结果；
- raw SPH / SPH_res / reconstruction / partial-correlation：无 S1/S2 OOF 结果；
- pCR complementarity 与 paired bootstrap：防火墙未开放；
- seed-level E1–E6：尚未产生。

## 下一步

等待至少一张 GPU 的空闲显存达到 60,000 MiB 后，按冻结配置运行 `scripts/run_representation_pipeline.py --mode run`；该入口会从当前已完成的 S0/residualizer 阶段安全续跑，依次执行 training、export、probe、aggregate、pCR-firewall audit 与 representation freeze。随后才可运行 `scripts/run_postfreeze_pipeline.py --mode run --clinical-table <private input>`。不得调整目标、lambda、checkpoint selector 或 Gate 阈值；不得在 `representation_freeze.json` 生成前运行 pCR evaluator。正式运行完成后，[decision.json](../metrics/decision.json) 将由 Gate A–D 的实际结果替换当前资源状态记录。
