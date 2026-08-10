# Eligible C1B-H model-input cache validation

本轮只对 technical-eligibility runner 产生的 947 名 eligible patients（3788 visits）执行冻结 C1B-H production builder。947/947 patient caches 已完成 schema-3 原子写入/复用、完整当前 input-contract rebuild、reload 与 model-only loader round-trip。

- exact eligibility valid-source count match：947/947 patients（全部四访）；
- frozen grid center match：947/947；
- finite/nonconstant/phase/shape/schema/provenance：947/947；
- cache completion：947/947 = 100%；
- model loader 仅返回 `[4,7,112,176,160] float32` DCE7，geometry/valid-source/support/phase/provenance 均为 sidecar。

此 cache 子门 PASS 仍不单独授权 Stage B；必须由 15 项 Stage-A finalizer 写出唯一 `STAGE_A_GO.json`。
