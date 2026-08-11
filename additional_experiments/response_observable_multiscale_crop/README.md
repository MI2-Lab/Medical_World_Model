# Response-Observable Multiscale Crop 试验记录

本目录保存 outcome-free Stage A 的公开试验记录，包括实验计划、冻结配置、实现代码、聚合指标、12 张审计图、provenance 与中文报告。最终判断为 `INPUT-CONTRACT PARTIAL`；Stage B、FTV+LD dual grounding、transition、clinical/treatment/pCR supervision 均未执行。

## 数据归属与许可证

本实验使用 The Cancer Imaging Archive（TCIA）公开的 I‑SPY2 / ACRIN-6698 影像资源。TCIA 当前数据页将相关影像及 Multi-feature MRI NACT Data 标注为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。使用本目录中的衍生图或数值时，请同时引用原始数据集：

- Li, W. et al. (2022). *I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2), Version 1*. The Cancer Imaging Archive. <https://doi.org/10.7937/TCIA.D8Z0-9T85>
- Newitt, D. C. et al. (2021). *ACRIN 6698/I-SPY2 Breast DWI*. The Cancer Imaging Archive. <https://doi.org/10.7937/TCIA.KK02-6D95>
- TCIA collection page: <https://www.cancerimagingarchive.net/collection/ispy2/>

图 10–12 是从公开 DCE-MRI 生成的去标识衍生图：执行物理空间中心平面采样、first-post-minus-pre 显示和审计用 FTV contour overlay；没有写入患者 ID 或源路径。其余图为聚合统计或示意图。本仓库及其作者不代表 TCIA 或原数据贡献者为这些衍生结果背书。

## 公开与私有边界

公开记录不包含原始 DICOM/NIfTI、模型权重、训练 feature、患者级标识或本机绝对路径。下列本地审计明细由 `.gitignore` 排除：

- `metrics/patient_visit_contracts.csv`
- `metrics/patient_level_geometry.csv`
- `metrics/patient_level_dicom_geometry.csv`
- preview cache、日志与 Python cache

公开 bundle 已通过 `scripts/verify_public_artifacts.py` 及其负例自测；真实图像还完成了本轮人工视觉复核。自动验证器不对任意未来 PNG 提供通用 OCR 保证。

## 导航

- [最终报告](reports/final_report.md)
- [病灶包含性审计](reports/containment_audit.md)
- [物理几何审计](reports/physical_geometry_audit.md)
- [DICOM 几何审计](reports/dicom_geometry_repair_audit.md)
- [图像质量与上下文分析](reports/image_quality_context_analysis.md)
- [实验计划](EXPERIMENT_PLAN.md)
- [Stage A gate](metrics/stage_a_gate.json)
