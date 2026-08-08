# 物理几何审计

## 结论

已完成375人、1,500个visit的shape、spacing、axis order、qform/sform、orientation及current crop physical FOV复核。历史reader的1,500个visit均采用index `XYZ`、无axis transpose，但它不读取affine，因此过去的shape/spacing match不能被解释为world-space registration。

- mask orientation计数：`{"LAS": 468, "RPS": 1032}`；同一患者四访方向一致。
- affine decision计数：`{"MASK_SFORM_GEOMETRY_CANDIDATE_REBUILD_DICOM_PIXELS": 37, "TRUST_DCE_QFORM_SFORM_SINGULAR": 35, "TRUST_DCE_SFORM": 1428}`。
- 当前仍沿native/T0 index basis采样，尚未执行跨患者anatomical orientation canonicalization。LAS/RPS等方向虽已审计，却不能仅凭common spacing视为common anatomical orientation；production必须冻结统一方向策略或验证native-orientation contract。
- DCE sform奇异：72/1,500；这些visit必须按raw DICOM重建pixel order后才可写model-ready cache。
- 当前header-only audit可计算physical containment，但model-ready geometry比例仅为95.2%；不能把mask sform candidate误称为已修复DCE image。
- obliquity>10°的QC flag为<5/1,500，最大12.50°。physical affine sampler支持正交oblique acquisition，因此这是QC/sensitivity flag，不把斜扫本身误判为几何失败。

## Spacing与legacy physical FOV

下表的IQR为Q25–Q75；spacing/FOV使用mm，shape与phase count为无量纲计数。

| scope | measurement | n | median | q25 | q75 | minimum | maximum |
|---|---|---|---|---|---|---|---|
| OVERALL | shape_x | 1500 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| OVERALL | shape_y | 1500 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| OVERALL | shape_z | 1500 | 80.0000 | 78.0000 | 134.0000 | 64.0000 | 256.0000 |
| OVERALL | raw_dce_phase_count | 1500 | 7.0000 | 7.0000 | 8.0000 | 4.0000 | 11.0000 |
| OVERALL | spacing_x_mm | 1500 | 0.6641 | 0.6250 | 0.7031 | 0.3125 | 1.3281 |
| OVERALL | spacing_y_mm | 1500 | 0.6641 | 0.6250 | 0.7031 | 0.3125 | 1.3281 |
| OVERALL | spacing_z_mm | 1500 | 2.0000 | 1.2000 | 2.2000 | 0.8000 | 2.5000 |
| OVERALL | legacy_fov_x_mm | 1500 | 63.7536 | 60.0000 | 67.5000 | 30.0000 | 127.5000 |
| OVERALL | legacy_fov_y_mm | 1500 | 63.7536 | 60.0000 | 67.5000 | 30.0000 | 127.5000 |
| OVERALL | legacy_fov_z_mm | 1500 | 64.0000 | 38.4000 | 70.4000 | 25.5999 | 80.0000 |
| OVERALL | acquisition_fov_x_mm | 1500 | 170.6667 | 160.0000 | 190.0032 | 100.0000 | 430.0000 |
| OVERALL | acquisition_fov_y_mm | 1500 | 170.6667 | 160.0000 | 190.0032 | 100.0000 | 430.0000 |
| OVERALL | acquisition_fov_z_mm | 1500 | 160.0000 | 160.0000 | 160.8000 | 128.0000 | 246.4000 |
| T0 | shape_x | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T0 | shape_y | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T0 | shape_z | 375 | 80.0000 | 78.0000 | 134.0000 | 64.0000 | 256.0000 |
| T0 | raw_dce_phase_count | 375 | 7.0000 | 7.0000 | 8.0000 | 5.0000 | 10.0000 |
| T0 | spacing_x_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.3125 | 1.3281 |
| T0 | spacing_y_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.3125 | 1.3281 |
| T0 | spacing_z_mm | 375 | 2.0000 | 1.2000 | 2.0500 | 0.8000 | 2.5000 |
| T0 | legacy_fov_x_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 30.0000 | 127.5000 |
| T0 | legacy_fov_y_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 30.0000 | 127.5000 |
| T0 | legacy_fov_z_mm | 375 | 64.0000 | 38.4000 | 65.6000 | 25.5999 | 80.0000 |
| T0 | acquisition_fov_x_mm | 375 | 175.0016 | 160.0000 | 190.0032 | 100.0000 | 420.0000 |
| T0 | acquisition_fov_y_mm | 375 | 175.0016 | 160.0000 | 190.0032 | 100.0000 | 420.0000 |
| T0 | acquisition_fov_z_mm | 375 | 160.0000 | 160.0000 | 160.8000 | 134.0000 | 224.0000 |
| T1 | shape_x | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T1 | shape_y | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T1 | shape_z | 375 | 80.0000 | 76.0000 | 134.0000 | 64.0000 | 256.0000 |
| T1 | raw_dce_phase_count | 375 | 7.0000 | 7.0000 | 8.0000 | 4.0000 | 11.0000 |
| T1 | spacing_x_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4492 | 1.3281 |
| T1 | spacing_y_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4492 | 1.3281 |
| T1 | spacing_z_mm | 375 | 2.0000 | 1.2000 | 2.2000 | 0.8000 | 2.5000 |
| T1 | legacy_fov_x_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 43.1232 | 127.5000 |
| T1 | legacy_fov_y_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 43.1232 | 127.5000 |
| T1 | legacy_fov_z_mm | 375 | 64.0000 | 38.4000 | 70.4000 | 25.5999 | 80.0000 |
| T1 | acquisition_fov_x_mm | 375 | 173.9130 | 160.0000 | 190.0032 | 114.9952 | 420.0000 |
| T1 | acquisition_fov_y_mm | 375 | 173.9130 | 160.0000 | 190.0032 | 114.9952 | 420.0000 |
| T1 | acquisition_fov_z_mm | 375 | 160.0000 | 160.0000 | 160.8000 | 143.0000 | 224.0000 |
| T2 | shape_x | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T2 | shape_y | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T2 | shape_z | 375 | 80.0000 | 80.0000 | 134.0000 | 64.0000 | 256.0000 |
| T2 | raw_dce_phase_count | 375 | 7.0000 | 7.0000 | 8.0000 | 5.0000 | 10.0000 |
| T2 | spacing_x_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4688 | 1.3281 |
| T2 | spacing_y_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4688 | 1.3281 |
| T2 | spacing_z_mm | 375 | 2.0000 | 1.2000 | 2.0500 | 0.8000 | 2.5000 |
| T2 | legacy_fov_x_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 45.0048 | 127.5000 |
| T2 | legacy_fov_y_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 45.0048 | 127.5000 |
| T2 | legacy_fov_z_mm | 375 | 64.0000 | 38.4000 | 65.6000 | 25.5999 | 80.0000 |
| T2 | acquisition_fov_x_mm | 375 | 170.0096 | 160.0000 | 188.7456 | 129.9968 | 420.0000 |
| T2 | acquisition_fov_y_mm | 375 | 170.0096 | 160.0000 | 188.7456 | 129.9968 | 420.0000 |
| T2 | acquisition_fov_z_mm | 375 | 160.0000 | 160.0000 | 160.8000 | 139.2000 | 246.4000 |
| T3 | shape_x | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T3 | shape_y | 375 | 256.0000 | 256.0000 | 256.0000 | 256.0000 | 384.0000 |
| T3 | shape_z | 375 | 80.0000 | 78.0000 | 134.0000 | 64.0000 | 252.0000 |
| T3 | raw_dce_phase_count | 375 | 7.0000 | 7.0000 | 8.0000 | 4.0000 | 10.0000 |
| T3 | spacing_x_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4688 | 1.3281 |
| T3 | spacing_y_mm | 375 | 0.6641 | 0.6250 | 0.7031 | 0.4688 | 1.3281 |
| T3 | spacing_z_mm | 375 | 2.0000 | 1.2000 | 2.2000 | 0.8000 | 2.5000 |
| T3 | legacy_fov_x_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 45.0048 | 127.5000 |
| T3 | legacy_fov_y_mm | 375 | 63.7536 | 60.0000 | 67.5000 | 45.0048 | 127.5000 |
| T3 | legacy_fov_z_mm | 375 | 64.0000 | 38.4000 | 70.4000 | 25.5999 | 80.0000 |
| T3 | acquisition_fov_x_mm | 375 | 170.0096 | 160.0000 | 182.5056 | 120.0128 | 430.0000 |
| T3 | acquisition_fov_y_mm | 375 | 170.0096 | 160.0000 | 182.5056 | 120.0128 | 430.0000 |
| T3 | acquisition_fov_z_mm | 375 | 160.0000 | 160.0000 | 160.8000 | 128.0000 | 246.4000 |

固定`32×96×96 voxel`并不对应固定物理视野。总体median约为X=63.75、Y=63.75、Z=64.00 mm；范围分别为30.00–127.50、30.00–127.50、25.60–80.00 mm。

## Full-support physical bbox extent

下表来自完整FTV inclusion support的source-domain physical bbox；它是保守proxy，远端碎片/多灶会放大extent，不等同radiologist LD target。

| axis | visit_adaptive_bbox_extent_median_mm | visit_adaptive_bbox_extent_q95_mm | visit_adaptive_bbox_extent_q99_mm | visit_adaptive_bbox_extent_max_mm |
|---|---|---|---|---|
| X | 47.0792 | 96.6736 | 122.1412 | 169.5313 |
| Y | 57.1621 | 108.9131 | 130.0056 | 177.1875 |
| Z | 54.0000 | 110.0000 | 134.0000 | 151.1996 |

## Visit-to-visit spacing variation

| axis | patients | patients_with_visit_variation | fraction_with_visit_variation | range_median_mm | range_q95_mm | range_max_mm | max_to_min_ratio_q95 | max_to_min_ratio_max |
|---|---|---|---|---|---|---|---|---|
| X | 375 | 139 | 0.3707 | 0.0000 | 0.1042 | 0.2783 | 1.1666 | 1.8750 |
| Y | 375 | 139 | 0.3707 | 0.0000 | 0.1042 | 0.2783 | 1.1666 | 1.8750 |
| Z | 375 | 17 | 0.0453 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.4545 |

## Physical frame限制

T0-anchored结果使用validated/repaired-header RAS-mm frame，但没有把T1–T3做image-only rigid registration。它能审计crop-induced recentering与coverage，却不能把patient repositioning和biological lesion displacement分开。所有后续报告均以`HEADER_PHYSICAL_UNREGISTERED`标记这一限制。
