# Four-Visit Valid-Source-Overlap Technical Eligibility Amendment

## 结论

这是在独立 provenance audit 判定 `AUDIT-NOT-REPAIRABLE` 后启动的**新预注册 run**，不是对旧 `STAGE_A_NO_GO` 的 post-hoc 修改。eligibility rule 已在本轮任何 cohort 结果、representation、training 或 FTV probe 之前冻结，plan SHA-256 为 `72c440302b3992a09bc74b0a6a59b8bf11d0e20835322f58cbbd5bd43608d588`。

通用程序从完整 candidate model-input population 机械运行 `all candidates → four-visit AND → eligible population`，未写入已知失败 patient、预期分母或目标排除数。实际结果为：

- candidate patients：948
- eligible patients：947
- excluded patients：1
- candidate visits：3792
- valid-source-overlap visits：3791
- zero-overlap visits：1
- exclusion reasons：`{"ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT": 1}`

Eligibility runner 只读取 imaging source、raw/rebuilt source geometry、预先物化的 frozen C1B-H physical grid 与 exact valid-source overlap。它没有读取 FTV、LD、SPH、BPE、lesion response、pCR、clinical、treatment、subtype、model loss、representation metric 或 downstream performance。

逐 patient/visit 的 ID、source path、affine 与 exact count 仅保存在 private manifests；公开产物只包含聚合计数和 SHA-256。旧 `STAGE_A_NO_GO` 与旧 `AUDIT-NOT-REPAIRABLE` 保持不可变。本 amendment 只确定新 Stage-A population，并不单独授权 Stage B。
