# Code/data-integrity bug-fix ledger

本 ledger 只记录本轮新实验中的必要 code/data-integrity 修正与执行异常；它不改变预注册科学问题、四臂定义、样本规则、阈值、模型结构或既有旧实验结论。冻结的 `EXPERIMENT_PLAN.md` SHA-256 为 `72c440302b3992a09bc74b0a6a59b8bf11d0e20835322f58cbbd5bd43608d588`。

## 1. 正式训练前：将非线性 SIGReg 恢复为 exact logical-batch B=32

- 阶段：Stage B 实现审计；发生在任何正式矩阵启动与科学结果之前。
- `src/c1b_stage_b/training.py`：`d1995d8df8ac01f1ab76efd328ba5ae700cb5b5577cde3ae12f1e358c764ab5b` → `2eb0f146bd5d69d0ddd2629ef7381e593703108e825e13fa739e350239b70cad`。
- 原因：旧原型按 physical microbatch 计算非线性 SIGReg；即使随后 accumulation，其梯度也不等于一次 B=32 SIGReg，违反冻结的 effective-batch contract。
- 修正：每个 logical 32-patient batch 先计算一次 SIGReg reference value/gradient，再对各 physical microbatch 使用数学等价 surrogate；每个 logical batch 仅执行一次 gradient clipping、optimizer step 与 EMA update。Validation 同样以 logical batch 聚合 SIGReg。
- 科学影响：无旧正式结果可保留或比较；旧 hash 下未启动正式矩阵。四臂架构、loss 权重、optimizer、LR、patient order 与训练预算均未改变。
- 重验证：4×8 与允许的全局 2×16 两种拆分对一次 B=32 reference gradient 的最大绝对误差均为 `5.96e-08`；Stage-B contract tests 与 full-epoch L1/L3 toy regression 均 PASS。

## 2. 首次正式启动的 first-batch fail-fast：no-grounding FTV count contract

- 阶段：首次 formal 4×8 启动；在第一个 L1 logical batch、任何 optimizer step、epoch checkpoint、history 或 selection 写出之前 fail-fast。
- `src/c1b_stage_b/training.py`：`2eb0f146bd5d69d0ddd2629ef7381e593703108e825e13fa739e350239b70cad` → `2edf546628e447bdd1b9715f60f105d1a5952763bd782aabddbae298fae62f52`。
- 原因：L1/N1 的冻结 objective 正确地报告零 grounded patients，但审计器错误地用共享数据中的 observable FTV metadata 作为它们的 expected count，导致零对非零的错误断言。
- 修正：L1/N1 的 expected logical grounded count 固定为零；L3/N3 仍逐 microbatch 汇总并与 logical observable-patient count 精确核对。
- 失败产物边界：`checkpoints/formal_4x8/` 被原样保留，仅含 `matrix_preflight.json` 与三个空 cell 目录；无 cell 文件，因此不进入任何训练、probe、aggregate 或科学结论。修正后从新的空根 `checkpoints/formal_4x8_restart1/` 完整重启 40-cell matrix，未 partial resume。
- 重验证：新增完整 L1 与 L3 toy epoch regression；两者均保持 exact B=32 gradient。新的正式矩阵在 4×8 下通过真实 N1 logical-batch resource smoke 后启动，未触发 2×16 fallback。

## 3. Stage A cache 复用时的 hardlink 执行异常（非科学代码修正）

- 事件：构建新 cache 时曾短暂为 262 个既有 cache 文件创建 hardlink。发现后，在 Stage-A finalizer 前逐个采用 copy/reflink、完整字节比较与 atomic replace 解除链接。
- 最终状态：新旧 cache 的 shared inode count 均为零；262/262 复用文件的 bytes、SHA-256、size 与 mtime 完全一致；新旧相关 cache 文件当前 link count 均为 1。旧冻结 sentinel、报告与 tracked-tree 内容 hash 未改变。
- 必须保留的限制：hardlink link-count 的创建/解除会更新 inode ctime，因此旧 cache inode ctime 可能改变；不得声称旧 cache 的所有 filesystem metadata 从未变化。
- 科学影响：无 tensor 字节、输入 contract、patient eligibility、Stage-A gate 数值或旧公开结论改变。机器可读证据为 `metrics/cache_independence_verification.json`。

## 4. 正式后处理前：fail-closed orchestration hardening

- 阶段：40-cell matrix 仍在训练、任何正式 feature/probe 输出之前。独立复审前的 `scripts/run_stage_b_postprocessing.py` SHA-256 为 `7e4176ffb14c40bdc2e716350d8e2eaf8e9796229375d02184262749f338824e`；最终为 `eedeb82ab1632a4342dfe536874bb943fef292a2b217f54a372dfb7600222981`。
- 问题与修正：补齐首个真实失败归因；加入 persistent `O_EXCL` claim 关闭并发 driver 对空 output root 的竞态；在 feature 与 probe 两阶段间重验 code SHA；加入 SIGINT/SIGTERM handler，确保 parent 被中断时回收 detached child process groups，避免中断后继续写正式产物。
- 科学影响：无。四臂模型、checkpoint、features、targets、Ridge 与统计实现均未改变；复审期间正式 feature/prediction roots 不存在，未运行真实数据后处理。
- 重验证：20/20 postprocess/contract synthetic tests 与 `py_compile` PASS；合成 SIGINT/SIGTERM 测试在约 0.25 秒内终止 active children，未留 orphan；三 GPU 计划为 14/13/13 cells，且只有 40/40 feature metadata/hash 全部通过后才允许 probes。

## 5. 结果使用边界

只有 `checkpoints/formal_4x8_restart1/matrix_complete.json` 中的完整 40-cell matrix 才可进入 frozen feature export 与 probes。任何 failed/partial root 均不得混入聚合。旧 `STAGE_A_NO_GO` 与 `AUDIT-NOT-REPAIRABLE` 始终作为不可变上游事实，而非本 ledger 所能修改的对象。

## 6. 正式后处理与聚合前：completion-chain、统计网格与事务发布 hardening

- 阶段：正式 40-cell matrix 训练期间、任何正式 feature/probe/aggregate 执行之前。未修改训练代码、正式 checkpoint、`EXPERIMENT_PLAN.md`、GO sentinel 或既有旧工件。
- 实现 SHA-256：
  - `src/c1b_stage_b/analysis.py`：`ee4dc23dbfc175b0912872910ab369d67d1f216707f2e6ce3f2bdabbdb209147` → `c0a69ca31bda8907b807c8804d6abcd66f90cfad3f8f715a808305565f2a8298`；
  - `scripts/aggregate_stage_b.py`：`1fbf0b53b981e35fc2899c685bb4c5c25a1ff727d70b39839390eb18e05ad2a7` → `4c9f89b801a89b68552898218fbffaee84aaaf717292aa066b511b96ef4623d5`；
  - `scripts/run_stage_b_postprocessing.py`：`eedeb82ab1632a4342dfe536874bb943fef292a2b217f54a372dfb7600222981` → `ed646113a1270179153574ab643cc8379ee2f00ae2e534949592e6d037c7f5b6`。
- 测试 SHA-256：
  - 新增 `tests/test_stage_b_aggregate.py`：`N/A` → `1ac5883de42407d6bccd99f950f47b9ed1fc8fa277d8b35df1f58b7945b4f28c`；
  - `tests/test_stage_b_contracts.py`：`4439c9080c0cecbcebec34870a6307af40f927b28d50ba4082a3f8b8791916dc` → `d8e2cfff4e79e9f5b402cbfabd9b442a6c1d94e8a08afb9998abdc6954984dfe`；
  - `tests/test_stage_b_postprocess.py`：`8322da1d0799f5ac59b964b909133d4bfcc66bffe00a8db55a21cd174135295d` → `0786358a720636f332ca4c0c206afa924ee97fce881e95557ee9aef5729f21dd`。
- 原因：预正式复核发现旧聚合器未机械绑定 `matrix_complete.json`、`postprocessing_complete.json`、data-contract SHA、正式 restart tag 与 40 个 metadata hash；对完全缺失的 task/seed/fold 组合可空通过；正式表图逐文件直写；Figure 6 仅显示 N3 并混合 seed/endpoint；浮点表示还可能把数学上恰为 5% 的 safety boundary 误判为超限。进一步对抗复核发现 selection/history 虽在后处理开始时验证，却未被 completion 链 hash 锚定，而它们会直接驱动 Table 4、optimization DiD/safety 与 Figure 12。
- 修正：
  - 聚合入口严格限定 `formal_4x8_restart1` 的 checkpoint/feature/prediction roots，重算 data-contract、matrix、claim、preflight、feature completion、postprocessing completion、40 个 feature/probe metadata 及其输出绑定；聚合前后各验证一次；
  - postprocessing 将 aggregate CLI 纳入 code inventory，并新增精确 40-cell selection/history SHA map 与 canonical digest；该证据贯穿 claim、preflight、feature completion 和 final completion，且在 feature/probe 两阶段后重新核验。selected checkpoint 继续由 feature metadata SHA 链绑定；
  - 强制 raw metrics 1,440-row、pooled metrics 144-row 的精确 Cartesian contract、每个 pooled group 五折覆盖及四臂 paired OOF；正式表固定 schema/行数：Table 2=`440`、Table 3=`32`、Table 4=`40`、Table 5=`10`、fold sensitivity=`720`；
  - 聚合采用 persistent `O_EXCL` claim、隐藏 staging、逐文件 exclusive atomic link、异常 rollback，并最后发布 summary commit marker；summary 记录所有 input completions、selection/history inventory、Table 2–5、Figure 4–12 与实现代码 SHA；
  - Figure 6 改为四臂 shared-axis panels，endpoint 分色、seed 分 marker/line，并按 seed 汇总五折 OOF calibration；新增 descriptive slope/intercept/mean-bias，结合 natural R² 与 prediction/target variance ratio 解释；
  - safety 判定直接比较 `selected_state_loss <= 1.05 * paired_baseline_state_loss`，保留原 degradation fraction 报告，因此精确 5% inclusive。
- 科学影响：无正式结果被重算、查看或替换；修正只加强预先冻结的完整性、发布与描述性 calibration 证据。四臂、样本、训练、Ridge selection、主要指标、DiD 定义与 5% 阈值均未改变；calibration slope/intercept/mean-bias 明确标记为 descriptive，而非新增选择终点。Macro 固定记录为 endpoint metric 的非加权均值，`n_test` 为 endpoint observations 之和而非独立患者数。
- 重验证：bowen 环境下新实验完整 test suite `46/46 PASS`（9.035 秒）；相关实现与测试 `py_compile` PASS；aggregate CLI `--help` PASS，并要求显式 `--execute`，无该参数仅校验 completion chain。40-cell E2E、tamper、缺整组、5% boundary、Figure 6、O_EXCL/staging failure 测试均只写入 `TemporaryDirectory`。未运行正式 aggregate 或 postprocessing。
