# Data Processing

This module documents and orchestrates the data-processing layer for the
I-SPY longitudinal DCE-MRI project.

The main goal is to make the data provenance clear:

1. Raw I-SPY2 and I-SPY1 DICOMs are converted to per-patient, per-visit 4D DCE
   NIfTI files.
2. I-SPY2 FTV analysis masks are converted and used for tumor localization and
   crop construction.
3. Clinical labels and MRI-NACT tabular features are extracted into CSV/JSON
   files under the preprocessing root.
4. DCE temporal phase choices are audited against BreastDCEDL metadata.
5. Modeling code builds DCE8 tensors/caches from the stored 4D DCE volumes and
   masks.

## Path Configuration

The clean code does not hard-code private machine paths. Configure paths with
CLI flags, environment variables, or an env file.

Required path variables:

```text
ISPY2_RAW_ROOT              I-SPY2 raw DICOM root
ISPY1_RAW_ROOT              I-SPY1 raw DICOM root
ISPY2_PREPROCESSED_ROOT     output root for preprocessed I-SPY2 data
ISPY1_PREPROCESSED_ROOT     output root for preprocessed I-SPY1 data
DCM2NIIX                    dcm2niix executable
BREASTDCEDL_METADATA_CSV    BreastDCEDL phase/crop metadata CSV
```

The preprocessing source code is included in this clean tree:

```text
ispy_jepa_tmi_clean/data_processing/preprocessing
```

The runner uses that copied directory by default. Override it only if you are
debugging against another preprocessing implementation.

Use `config/paths.example.env` as the template for a machine-local config file:

```bash
cp ispy_jepa_tmi_clean/data_processing/config/paths.example.env \
  ispy_jepa_tmi_clean/data_processing/config/paths.local.env
```

Then edit `paths.local.env` for that machine.

## Shared Paths On This Server

On the current lab server, raw and already preprocessed data are stored in
shared `/data` locations. Use:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py \
  --env-file ispy_jepa_tmi_clean/data_processing/config/paths.shared-data.env \
  --stage check
```

The shared data config contains:

```text
ISPY2_RAW_ROOT=/data/data/Breast_Cancer/I-SPY2
ISPY1_RAW_ROOT=/data/data/Breast_Cancer/I-SPY1
ISPY2_PREPROCESSED_ROOT=/data/data/Preprocessed/I-SPY2
ISPY1_PREPROCESSED_ROOT=/data/data/Preprocessed/I-SPY1
```

This shared config intentionally does not include `/home/<user>` paths. Directly
using the already processed NIfTI files, clinical tables, MRI-NACT feature
tables, and modeling caches only needs the preprocessed roots above. Rerunning
DICOM conversion additionally requires `DCM2NIIX`; rerunning the BreastDCEDL
timing audit additionally requires `BREASTDCEDL_METADATA_CSV`.

## Clean Entry Point

All stages are launched through:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py --stage check
```

With a local env file:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py \
  --env-file ispy_jepa_tmi_clean/data_processing/config/paths.local.env \
  --stage check
```

By default the runner is a dry run. Add `--execute` to actually run a stage:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py \
  --env-file ispy_jepa_tmi_clean/data_processing/config/paths.local.env \
  --stage clinical \
  --execute
```

Available stages:

```text
check             Print path and output-file status.
ispy2-dicom       Convert all I-SPY2 DICOM data to per-visit NIfTI manifests.
ispy1-dicom       Convert all I-SPY1 DICOM data to per-visit NIfTI manifests.
clinical          Extract I-SPY2 clinical labels.
mri-nact          Extract I-SPY2 MRI-NACT tabular features.
dce-timing-audit  Audit DCE frame timing and phase-index policy.
all               Run the full data-processing sequence above.
```

Useful options:

```text
--workers N       Number of workers for batch DICOM conversion/audit.
--limit N         Limit patients for a smoke test where supported.
--overwrite       Rebuild outputs where the source script supports it.
--execute         Run commands. Without this flag, commands are printed only.
```

Example smoke test:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py \
  --env-file ispy_jepa_tmi_clean/data_processing/config/paths.local.env \
  --stage ispy2-dicom \
  --limit 2 \
  --workers 2 \
  --execute
```

## Output Structure

Each I-SPY2 patient has:

```text
<ISPY2_PREPROCESSED_ROOT>/<patient_id>/
  manifest.json
  visit_summary.csv
  T0/
    <patient_id>_T0_original_DCE.nii
    <patient_id>_T0_original_DCE.json
    <patient_id>_T0_analysis_mask_raw.nii
    <patient_id>_T0_analysis_mask_raw.json
    <patient_id>_T0_ftv_mask.nii
  T1/
  T2/
  T3/
```

Each I-SPY1 patient has:

```text
<ISPY1_PREPROCESSED_ROOT>/<patient_id>/
  manifest.json
  T0/
    <patient_id>_T0_original_DCE.nii
    <patient_id>_T0_original_DCE.json
  T1/
  T2/
  T3/
```

I-SPY1 currently does not provide the same FTV analysis-mask files in this
preprocessed output.

## I-SPY2 Analysis Mask Meaning

The I-SPY2 analysis-mask files are not ordinary binary tumor segmentation masks.
They come from the FTV analysis workflow:

- voxels with value `0` are included in the FTV/tumor analysis region;
- nonzero values are exclusion labels/bits from the FTV analysis mask.

In this project we therefore use the analysis-mask-derived FTV mask primarily
for localization, bounding boxes, and tumor-centered crops. It should not be
described as a manual dense lesion segmentation.

## DCE Storage Versus DCE8 Modeling Tensor

The preprocessing stage stores the full 4D DCE NIfTI for every visit:

```text
original_DCE.nii shape: X x Y x Z x T
```

The compact DCE8 tensor is derived later for modeling. DCE8 is not eight
independent acquisition channels. It is a per-visit summary built from selected
DCE phases:

```text
channel 0: pre-contrast
channel 1: early post-contrast
channel 2: late post-contrast
channel 3: early - pre
channel 4: late - pre
channel 5: peak relative enhancement
channel 6: washout relative enhancement
channel 7: analysis-mask / localization channel
```

The relative enhancement channels use the pre-contrast image as denominator
with numerical clipping in the modeling dataset code.

## DCE Phase Policy

The current recommended wording is:

```text
metadata-aware early/late phase selection with short-series fallback
```

Do not describe this as a vague adaptive rule without defining it. The exact
policy used in the current best image pipeline is:

1. Read `pre`, `post_early`, and `post_late` from the BreastDCEDL metadata when
   available.
2. Clip every index to the available DCE frame range.
3. For visits with more than four frames, use metadata `post_early` and
   `post_late`.
4. For short DCE series with four or fewer frames, use the first post-contrast
   frame as early and the last frame as late.
5. The peak-enhancement channel is computed from all available post-contrast
   frames.

This replaced the earlier simple policy `early = frame 1, late = last frame`.
The timing audit showed that `frame 1` was usually much earlier than the
BreastDCEDL early phase, and the last frame was often later than the metadata
late phase.

## Main Generated Tables

I-SPY2 clinical labels:

```text
<ISPY2_PREPROCESSED_ROOT>/clinical_labels.csv
<ISPY2_PREPROCESSED_ROOT>/clinical_labels_complete4visits.csv
<ISPY2_PREPROCESSED_ROOT>/clinical_label_dictionary.json
```

I-SPY2 MRI-NACT features:

```text
<ISPY2_PREPROCESSED_ROOT>/mri_nact_features_wide.csv
<ISPY2_PREPROCESSED_ROOT>/mri_nact_features_long.csv
<ISPY2_PREPROCESSED_ROOT>/mri_nact_features_complete4visits_wide.csv
<ISPY2_PREPROCESSED_ROOT>/mri_nact_features_with_clinical_labels.csv
<ISPY2_PREPROCESSED_ROOT>/mri_nact_feature_dictionary.json
```

I-SPY1 labels:

```text
<ISPY1_PREPROCESSED_ROOT>/clinical_labels_complete4visits.csv
```

## Current Modeling Cache To Know About

The current best mixed I-SPY2/I-SPY1 DCE8 cache is:

```text
<ISPY2_PREPROCESSED_ROOT>/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96
```

This cache is a modeling artifact, not the raw preprocessing output. It is
derived from the `original_DCE.nii` files and localization masks described
above.
