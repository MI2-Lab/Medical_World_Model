# Anatomical orientation validation

## 结论

全部 3,792 个实际model-input visit（948 人）的空间header均可先依据 affine 对 array 执行真实 permutation/flip，再进入 RAS+ physical sampling；不是只改 orientation label。146 个 I-SPY2 singular-sform visit和560个strict-eligible I-SPY1 visit均使用已经逐像素验收的 raw-DICOM rebuilt volume；未通过source/phase/pixel硬门的I-SPY1患者不进入该population。

- resolved input orientation：`{"LAS": 1021, "LIP": 1, "LPS": 146, "PIR": 559, "RPS": 2065}`；输出统一为 `RAS+`：3,792/3,792。
- canonical footprint round-trip最大误差：8.04e-14 mm。
- DCE-mask footprint corner最大误差：0.001836 mm（门槛 0.1 mm）。
- 输出轴严格按 R/A/S 正方向；DCE和localization support分别按自身 affine 重排后在同一RAS物理坐标采样。
- production builder和单元测试会检查真实数组内容随 affine permutation/flip 一起重排；geometry metadata与valid-source mask均为sidecar，不进入DCE7 tensor。

因此 orientation 子门：**PASS**。
