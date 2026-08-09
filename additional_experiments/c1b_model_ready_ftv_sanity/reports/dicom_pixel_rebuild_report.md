# Raw-DICOM PixelData 重建报告

## 结论

正式 375 人范围内要求的 72 个 singular-sform visit：**72/72 PASS**。为了不让 matched base-training population 的新输入臂接收坏 geometry，另对 375 人之外发现的 74 个 visit 应用完全相同的 fail-closed rebuild：**74/74 PASS**。综合 gate：**PASS**。

## 验收

| scope | visits | pass | fail | DICOM cells | verified cells | max cell error | max footprint corner error (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 正式72 | 72 | 72 | 0 | 77792 | 77792 | 0.0 | 2.1864245105181427e-05 |
| base-only扩展74 | 74 | 74 | 0 | 75320 | 75320 | 0.0 | 2.4654460230326754e-05 |
| 全部 | 146 | 146 | 0 | 153112 | 153112 | 0.0 | 2.4654460230326754e-05 |

每个 series 均要求完整且唯一的 TPI/AcquisitionTime x IPP-slice cell、逐文件 scaling、finite/nonconstant float32 volume、第二次独立 PixelData decode 后逐 cell exact compare、与 reference mask 的 center/footprint corner误差不超过0.1 mm，并在写出后验证 qform/sform。患者级路径、UID、cell hash和输出 hash只保存在 gitignored private sidecar；公开文件无患者身份。
