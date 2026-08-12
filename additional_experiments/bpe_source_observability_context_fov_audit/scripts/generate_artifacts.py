#!/usr/bin/env python3
"""Generate the aggregate figure and Chinese reports from audited metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    rows = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def generate_figure() -> None:
    extent = pd.read_csv(ROOT / "metrics" / "acquisition_image_support_extent.csv")
    values = {
        "F0 LOCAL\nreadout": np.asarray([64.0, 64.0, 64.0]),
        "F1 Full\nC1B-H": np.asarray([144.0, 158.4, 224.0]),
    }
    for statistic in ("Q50", "Q95", "Q99"):
        subset = extent.loc[extent["statistic"].eq(statistic)].set_index("axis")
        values[f"F2 recon AABB\nmarginal {statistic}"] = np.asarray(
            [subset.loc[axis, "extent_mm"] for axis in ("X", "Y", "Z")]
        )

    labels = list(values)
    matrix = np.stack([values[label] for label in labels])
    axes = ("X", "Y", "Z")
    colors = ("#3973ac", "#e07a3f", "#5b9a6f")
    x = np.arange(len(labels), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.22)
    for index, (axis, color) in enumerate(zip(axes, colors, strict=True)):
        bars = ax.bar(
            x + (index - 1) * width,
            matrix[:, index],
            width,
            label=f"{axis} extent",
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in matrix[:, index]], padding=2, fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Physical extent (mm)")
    fig.suptitle(
        "BPE context-FOV geometry audit: support extent is known, source coverage is not",
        y=0.98,
        fontsize=16,
    )
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, loc="upper left")
    fig.text(
        0.5,
        0.885,
        "SOURCE ROI NOT AVAILABLE\nOccupancy / margin / boundary touch: NOT EVALUABLE",
        ha="center",
        va="center",
        fontsize=10,
        color="#8f1d1d",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff1f1", "edgecolor": "#c86464"},
    )
    fig.text(
        0.5,
        0.045,
        "F0 is a readout support and still consumes the full F1 encoder input.\n"
        "F2 bars are separate marginal RAS AABB summaries of reconstructed source-image support; "
        "they are not one tensor, raw-footprint equivalence, breast/BPE coverage, or CXT2.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    output = ROOT / "figures" / "fov_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Software": "matplotlib", "Title": "BPE FOV comparison"})
    plt.close(fig)


def generate_source_contract() -> None:
    config = load_json(ROOT / "configs" / "audit.json")
    source = config["source_contract"]
    report = f"""# BPE source contract

## 结论

本实验确认 BPE 的全称是 **background parenchymal enhancement（背景实质增强）**。
真实字段是 `BPE_5slice_mean_T0` 至 `BPE_5slice_mean_T3`。其 source semantics 可以由
原始方法论文和冻结 feature inventory 恢复，但生成这些标量时的真实 breast/FGT mask、
五层坐标与 patient-specific laterality mapping 没有随 workbook 或现有 manifests 发布。

**Source status：`SOURCE_ROI_NOT_AVAILABLE`。** 本实验没有、也不得用新分割或半乳 volume
伪造 historical source ROI。

## BPE_source_semantics

| 项目 | 冻结定义 | 证据状态 |
|---|---|---|
| 全称 | background parenchymal enhancement | confirmed |
| 乳腺侧 | 原发肿瘤/病灶的对侧乳腺（contralateral） | method confirmed；patient-level L/R mapping unavailable |
| slice | superior–inferior 方向几何居中的连续 5 个 axial slices | method confirmed；真实 indices unavailable |
| tissue | 自动分割乳腺中的 fibroglandular tissue（FGT） | method confirmed；真实 mask unavailable |
| intensity | early percent enhancement：precontrast 到 early postcontrast；不是 late PE 或 SER | method confirmed |
| statistic | 五层 FGT voxels 的 mean early PE | method confirmed |
| lesion mask | 不用于定义/测量 BPE voxels | method confirmed |
| lesion laterality | 选择 contralateral side 所必需 | required but mapping unavailable |
| ROI method | automatic breast-boundary segmentation + fuzzy c-means FGT classification | algorithmic, not manual |
| timing | T0、T1、T2、T3 各有 visit-specific scalar | confirmed |
| ROI temporal anchor | 未说明每次重分割、T0 propagation 或 longitudinal registration | not documented |
| workbook unit | 字段名/方法是 mean PE；workbook 本身没有独立 unit metadata | qualified |

方法上的 early PE 为 `(S_early - S_pre) / S_pre × 100%`。原论文将 early acquisition
描述为注射后约 2.5 分钟；late acquisition 参与 SER，但不属于这个 BPE 标量。

## BPE_source_geometry_requirements

若要可靠重跑 F0/F1/F2 observability gates，至少必须取得每个 patient × visit 的：

1. 生成 workbook BPE 时的真实 contralateral breast boundary；
2. 真实 five-slice indices 或对应 physical slab；
3. 真实 FGT voxel mask/contours，而不是后验重建的 proxy；
4. lesion side 与 contralateral side 的 hash-bound mapping；
5. source image/phase UID、DICOM LPS geometry、NIfTI/RAS affine；
6. segmentation 是否在 image boundary 被截断，以及需要的 pre/early phase common support；
7. ROI 是逐访视重分割、baseline anchored，还是通过何种 registration 传播。

映射必须沿 `raw DICOM LPS → reconstructed NIfTI → true array-reordered RAS+ → frozen
C1B-H grid` 执行，并以 voxel-center affine 和完整 half-voxel footprint 计算 physical
intersection。不能只算 array-index overlap；reflect padding 也不能算 valid source。

## 已恢复与未恢复的 evidence

- Source workbook：384 人，四个 absolute BPE 字段均 finite；workbook 只有一个 ID、16 个
  absolute scalars 和 12 个 derived changes，没有 mask、坐标、laterality 或 affine。
- Matched audit cohort：375 人 × 4 visits；另外 9 个 workbook patients 因没有完整四访视
  C1B match 被技术性排除。
- Raw/reconstructed support：1,500/1,500 selected DICOM series 与 reconstructed DCE 可用；
  375/375 frozen C1B caches 可用。F2 extent 取自 reconstructed source affine；本 audit
  不从 full raw series 独立重算 footprint equivalence。
- Existing masks 是 lesion/FTV analysis masks，不能替代 contralateral FGT source ROI。
- 先前 FOV firewall 也把 ROI/laterality mapping、overlap 和 boundary audit 记为 unavailable。

## Evidence provenance

- 方法论文：[Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI](https://doi.org/10.1038/s41523-020-00203-7)
- Open-access full text：[PMC7695723](https://pmc.ncbi.nlm.nih.gov/articles/PMC7695723/)
- 冻结 target inventory：`additional_experiments/nonftv_phenotype_decodability_audit/manifests/target_contract.csv`
- 冻结 FOV firewall：`additional_experiments/nonftv_phenotype_decodability_audit/metrics/bpe_fov_observability_audit.csv`
- Goal 6 feature inventory commit：`{config['provenance']['goal6_source_commit_sha']}`
- Source workbook SHA-256：`{config['paths']['source_workbook_sha256']}`
"""
    output = ROOT / "reports" / "bpe_source_contract.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def generate_final_report() -> None:
    config = load_json(ROOT / "configs" / "audit.json")
    decision = load_json(ROOT / "metrics" / "decision.json")
    geometry = load_json(ROOT / "metrics" / "source_geometry_summary.json")
    grid = load_json(ROOT / "metrics" / "c1b_grid_validation.json")
    orientation = load_json(ROOT / "metrics" / "orientation_validation.json")
    population = pd.read_csv(ROOT / "manifests" / "audit_population.csv")
    coverage = pd.read_csv(ROOT / "metrics" / "coverage_table.csv")
    laterality = pd.read_csv(ROOT / "metrics" / "laterality_audit.csv").iloc[0]
    longitudinal = pd.read_csv(ROOT / "metrics" / "longitudinal_geometry.csv")
    cost = pd.read_csv(ROOT / "metrics" / "context_candidate_cost_table.csv")

    ext = geometry["reconstructed_source_image_aabb_extent_marginal_xyz_mm"]
    f1 = config["fov_contracts"]["F1"]
    f1_row = cost.loc[cost["contract"].eq("F1_FULL_C1B_H")].iloc[0]
    four_visit_memory = float(f1_row["float32_memory_mib_per_visit"]) * 4.0
    delivery = decision["delivery"]

    population_md = markdown_table(
        population,
        ["stage", "patients", "visits", "included", "reason"],
        ["Stage", "Patients", "Visits", "Included", "Reason"],
    )
    coverage_md = markdown_table(
        coverage,
        ["fov_contract", "eligible_visits", "source_roi_available_visits", "source_occupancy_evaluable_visits", "observability_gate", "status"],
        ["FOV", "Eligible visits", "ROI available", "Coverage evaluable", "Gate", "Status"],
    )
    longitudinal_md = markdown_table(
        longitudinal,
        ["interval", "eligible_patients", "evaluable_patients", "roi_temporal_anchor", "status"],
        ["Interval", "Eligible", "Evaluable", "ROI anchor", "Status"],
    )
    cost_md = markdown_table(
        cost,
        ["contract", "role", "extent_x_mm", "extent_y_mm", "extent_z_mm", "voxels_per_visit", "float32_memory_mib_per_visit", "relative_voxel_cost_vs_C1B", "status"],
        ["Contract", "Role", "X mm", "Y mm", "Z mm", "Voxels/visit", "MiB/visit", "vs C1B", "Status"],
    )

    report = f"""# BPE Source Observability & Context-FOV Audit 最终报告

## 执行结论

唯一合法 scientific classification 是 **D — `BPE_SOURCE_NOT_RELIABLY_AUDITABLE`**。
原因不是 selected-series、reconstructed-image 或 C1B-cache availability 缺失：375 人 ×
4 visits 的 1,500 个 selected raw DICOM series、1,500 个 reconstructed DCE 和 375 个
frozen C1B caches 均可用。
决定性缺口是生成 BPE 标量时的真实 contralateral breast/FGT source ROI、五层坐标和
hash-bound lesion→contralateral mapping 未发布。

因此 F0 LOCAL、F1 full C1B-H、F2 reconstructed source-image 三个 observability gates 都是
**`NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE`**。这不是三次 FAIL，也不证明 LOCAL/C1B
disjoint、acquisition limited 或 representation failure。下一阶段选择是
**`PAUSE_BPE`**；本实验 **不授权** Local–Context Phenotype Representation Pilot。

## Outcome-blind firewall 与 audit population

本运行没有读取 pCR、HR、HER2、treatment、clinical model performance；BPE 数值只在
读取 workbook 后立即归约为 finite/missing availability flag，没有被保留、排序或用于
case/FOV/gate selection。20 人 geometry sample 按 salted patient-ID hash 固定抽取，
selection SHA-256 为 `{laterality['selection_sha256']}`。

{population_md}

## 三个 input contracts 与 geometry validation

- **F0**：固定 `64 × 64 × 64 mm` LOCAL readout support。它不是 input crop；encoder
  仍读取完整 F1 tensor，不能把 receptive field 或 padding 当作 source observability。
- **F1**：固定 ZYX `{f1['shape_zyx']}`、XYZ spacing `{f1['spacing_xyz_mm']}` mm、
  XYZ footprint `{f1['extent_xyz_mm']}` mm、true RAS+、T0 anchor、不重心化。375/375
  affines/grid centers 验证通过，max center/extent error 都为
  `{grid['maximum_center_consistency_error_mm']:.3e}` mm。
- **F2**：reconstructed source affine 对 half-voxel bounds 定义的 source-image
  parallelepiped，并关联到 frozen selected raw series。Raw series existence 与 first-instance
  geometry tags 已核对，但没有从 full raw series 独立重建第二个 footprint；因此不声称
  raw↔reconstructed footprint equivalence。公开 extent 是各轴独立的 RAS AABB marginal
  summaries，不能用于代替 exact F2 containment，也不是一个 realizable median tensor。

Primary 1,500 visits 的 canonical orientation 前为
`{orientation['orientation_before']}`，后为 1,500/1,500 RAS+；max voxel-center corner
roundtrip error `{orientation['maximum_roundtrip_corner_error_mm']:.3e}` mm，DCE/mask
footprint error max `{orientation['maximum_dce_mask_footprint_corner_error_mm']:.6f}` mm。

## Primary observability results

{coverage_md}

Source occupancy、X/Y/Z boundary touch、centroid inclusion、physical margin、ROI 内
valid-source/padding/extrapolation 比例和 contralateral breast coverage 都需要真实 ROI。
现有 whole-grid C1B/source overlap（1,500/1,500 >0）只证明 reconstructed source image
与 C1B 有某些交集，不能替代 BPE ROI overlap。

## F2 reconstructed source-image support 描述（不是 BPE context FOV）

| Statistic | X AABB mm | Y AABB mm | Z AABB mm |
|---|---:|---:|---:|
| Q50 | {fmt(ext['x']['q50'])} | {fmt(ext['y']['q50'])} | {fmt(ext['z']['q50'])} |
| Q90 | {fmt(ext['x']['q90'])} | {fmt(ext['y']['q90'])} | {fmt(ext['z']['q90'])} |
| Q95 | {fmt(ext['x']['q95'])} | {fmt(ext['y']['q95'])} | {fmt(ext['z']['q95'])} |
| Q99 | {fmt(ext['x']['q99'])} | {fmt(ext['y']['q99'])} | {fmt(ext['z']['q99'])} |

X/Y/Z 的每个分位数都是单独计算的 marginal statistic；同一行不代表某一真实 visit 或
一个可实现的 tensor geometry。{geometry['source_affine_oblique_visits']}/1,500 source affines 是 oblique，因此 AABB 也不是 exact
parallelepiped。这些数只描述 reconstructed source image；没有 breast mask/FGT ROI 时，
不能证明 BPE source 完整、不能定义 bilateral crop，也不能被称为 CXT2 所需 FOV。

## Laterality / alignment audit

20 人 × 4 visits 的 outcome-blind sample 全部通过 RAS/affine consistency。全 1,500
visits 中 DICOM `Laterality` 有值 `{int(laterality['population_laterality_tag_present_visits'])}`，
缺失 `{int(laterality['population_laterality_tag_missing_visits'])}`；`ImageLaterality` 有值
`{int(laterality['population_image_laterality_tag_present_visits'])}`；
`{int(laterality['population_patients_with_conflicting_known_laterality_tag'])}` 人的已知
`Laterality` 跨 visits 冲突。IOP/IPP/PixelSpacing/Rows/Columns/FrameOfReferenceUID 在
`{int(laterality['population_geometry_tags_complete_visits'])}`/1,500 first-instance headers
齐全。

但 `Laterality` 不完整，且没有 hash-bound 到 lesion side 或 historical BPE source-side
selection；RAS X 正负也不能独自定义 patient midline。结论只能是
**geometry orientation passed，BPE contralateral mapping unresolved**。

## Longitudinal geometry

{longitudinal_md}

T0/T1/T2/T3 的 BPE scalar 是 visit-specific measurement；原方法没有说明真实 FGT mask
是每访视重分割、baseline anchored 还是注册传播。故 centroid displacement、physical
volume variation 与 laterality consistency 都不能用 acquisition center 或 lesion mask
代算。本实验也没有为了 temporal consistency 注册 future visit。

## Compute / resolution audit

{cost_md}

F1 为 `3,153,920` voxels/visit；DCE7 float32 原始 image tensor 为
`{float(f1_row['float32_memory_mib_per_visit']):.3f} MiB/visit`，四访视为
`{four_visit_memory:.3f} MiB/patient`，未计 gradients、activations 或 framework overhead。
F0 只是 readout，因此实际 input cost 仍为 F1 的 1.0×。64-mm support 若仅按 C1B spacing
离散化会是 `32×72×72=165,888` voxels，但这不是当前 model input。

F2 的 5,242,880 voxels/visit 与 20 MiB 是 source voxel count 各自 Q50 所导出的描述性
single-float32-volume lower bound；它与 X/Y/Z marginal Q50 不共同定义一个真实 visit，
也不是 DCE7 context tensor cost。由于 CXT2 没有被可靠定义，不能给 production tensor、
downsampling factor 或 `MULTISCALE_CONTEXT_RECOMMENDED`。

## 十四个问题逐项回答

### 1. BPE 真实来源是什么？

Background parenchymal enhancement：对侧乳腺在 S–I 方向几何居中的连续五个 axial
slices 中，自动 breast segmentation + fuzzy c-means 得到的 fibroglandular tissue 的
mean early percent enhancement。不是 ipsilateral lesion ROI、late PE、SER 或 manual ROI。

### 2. Source ROI 是否可恢复？

否。状态为 **`SOURCE_ROI_NOT_AVAILABLE`**。Scalar semantics 可恢复，historical voxel
source 不可恢复；不得伪造 proxy ROI。

### 3. LOCAL 覆盖多少？

**NA / NOT EVALUABLE**。64-mm support 已冻结，但没有 source ROI 可与其做 physical
intersection。

### 4. Full C1B 覆盖多少？

**NA / NOT EVALUABLE**。F1 geometry 完整且 1,500/1,500 visits 有某些 valid-source
overlap，但这不是 BPE ROI occupancy。

### 5. 原 acquisition 覆盖多少？

**NA / NOT EVALUABLE**。Reconstructed F2 source-image support 可恢复且 selected raw series
存在，但本 audit 未独立重建 raw full-series footprint equivalence；更关键的是 BPE source
completeness 不可恢复，image boundary 不能冒充 breast boundary。

### 6. 是否有 boundary-touch？

未知。F0/F1/F2 的 BPE source boundary-touch 均不可计算。

### 7. 是否有左右侧映射风险？

有。True RAS conversion 已通过，但 patient-specific lesion→contralateral source mapping
不存在；DICOM laterality tag 又不完整且非 authoritative BPE provenance。

### 8. Longitudinal geometry 是否稳定？

不可判断。三个 interval 的 centroid/volume/laterality rows 均 0 evaluable patients。

### 9. Full C1B 是否已经足够？

不可判断，不能归类为 A `FULL_C1B_CONTEXT_SUFFICIENT`。

### 10. 是否需要 bilateral broader context？

不可判断，不能归类为 B `BROADER_BILATERAL_CONTEXT_REQUIRED`；nominal anatomy 语义
不能替代 ROI coverage gate。

### 11. Broader context 大约需要多大物理 FOV？

当前不能定义。上表 F2 Q50/Q90/Q95/Q99 仅为 reconstructed source-image AABB 的独立
marginal statistics，不是所需 bilateral breast FOV、单一 tensor geometry 或 crop proposal。

### 12. 计算代价如何？

F0 实际仍使用 F1 input；F1 为 84.219 MiB/visit、336.875 MiB/four visits 的 DCE7
float32 image tensor。CXT2 未定义，故没有诚实的相对 production cost。

### 13. 下一阶段 phenotype branch 应该使用什么？

选择 **暂停 BPE（`PAUSE_BPE`）**，不是 current LOCAL、full C1B 或 bilateral context。
先恢复 authoritative source ROI、five-slice mapping 与 lesion-side provenance，再原样重跑
本实验 gates。

### 14. 本实验是否授权下一阶段 context representation pilot？

**否。** 只有分类 A 或 B 才授权 Local–Context Phenotype Representation Pilot；本实验
是 D。也不能跳到 phenotype-specific representation audit，因为那要求先证明 C：LOCAL
already observable。

## Scientific classification 与授权

- Classification：**D — `BPE_SOURCE_NOT_RELIABLY_AUDITABLE`**
- Next input：**`PAUSE_BPE`**
- Context candidate：`null`
- Local–Context Phenotype Representation Pilot：**NOT AUTHORIZED**
- Machine-readable decision：`metrics/decision.json`

## Git / delivery provenance

- Parent branch：`{config['provenance']['parent_branch']}`
- Parent commit：`{config['provenance']['parent_commit_sha']}`
- Start timestamp：`{config['provenance']['start_timestamp']}`
- Experiment branch：`{delivery['branch']}`
- Audit commit SHA：`{delivery['audit_commit_sha']}`
- Push status：`{delivery['push_status']}`

## 主要公开 artifacts

- Source contract：`reports/bpe_source_contract.md`
- Coverage：`metrics/coverage_table.csv`
- Boundary touch：`metrics/boundary_touch_table.csv`
- Physical margin：`metrics/physical_margin_distribution.csv`
- Laterality：`metrics/laterality_audit.csv`
- Longitudinal：`metrics/longitudinal_geometry.csv`
- FOV/cost：`metrics/acquisition_image_support_extent.csv`、
  `metrics/context_candidate_cost_table.csv`
- Figure：`figures/fov_comparison.png`
- Decision：`metrics/decision.json`
"""
    output = ROOT / "reports" / "final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def main() -> None:
    generate_figure()
    generate_source_contract()
    generate_final_report()
    print(json.dumps({"status": "COMPLETE", "reports": 2, "figures": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
