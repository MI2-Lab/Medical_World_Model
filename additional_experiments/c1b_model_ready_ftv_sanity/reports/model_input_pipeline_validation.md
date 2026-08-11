# Model-input pipeline validation

## 结论

完整的 948 人、3792 visit header-only overlap audit 在冻结人口中发现 1 个 `ZERO_VALID_SOURCE_OVERLAP` visit（valid-source voxel = 0）。因此 production-like cache validation **FAIL**；不得改变冻结人口、不得继续 full-scope cache、不得启动 Stage B。

- 本次 validation 目标为 263 人；已有 262/263 个患者 cache 通过原子写入，但 cohort-level schema-3 cache contract 未闭合，不能据此判 PASS。
- 失败病例标识与路径只存在 private failure table/selection manifest；公开报告仅保留计数与 SHA-256 闭包。
- 未放宽阈值，也未事后排除病例。
