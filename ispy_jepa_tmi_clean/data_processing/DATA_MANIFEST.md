# Data Manifest

This file lists the data products used by the project and the path variables
that locate them on a given machine. It deliberately avoids hard-coded private
server paths.

## Shared Paths On Current Lab Server

Users on the same lab server can use the already processed data from shared
`/data` locations:

```text
ISPY2_RAW_ROOT=/data/data/Breast_Cancer/I-SPY2
ISPY1_RAW_ROOT=/data/data/Breast_Cancer/I-SPY1
ISPY2_PREPROCESSED_ROOT=/data/data/Preprocessed/I-SPY2
ISPY1_PREPROCESSED_ROOT=/data/data/Preprocessed/I-SPY1
```

The corresponding env file is:

```text
ispy_jepa_tmi_clean/data_processing/config/paths.shared-data.env
```

No `/home/<user>` paths are required to read the processed data. `DCM2NIIX`
must be configured only when rerunning DICOM conversion. The BreastDCEDL phase
metadata used by the model cache is included under `data_processing/metadata`.

## Raw Data

```text
<ISPY2_RAW_ROOT>
<ISPY1_RAW_ROOT>
```

Important I-SPY2 spreadsheets:

```text
<ISPY2_RAW_ROOT>/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx
<ISPY2_RAW_ROOT>/Multi-feature-MRI-NACT-Data.xlsx
<ISPY2_RAW_ROOT>/Analysis-mask-files-description.v20211020.docx
```

## DICOM Converter

```text
<DCM2NIIX>
```

The clean runner prepends the parent directory of `<DCM2NIIX>` to `PATH` and
sets `DCM2NIIX` before launching the preprocessing scripts.

## Preprocessed I-SPY2 Root

```text
<ISPY2_PREPROCESSED_ROOT>
```

Top-level files:

```text
_batch_summary.csv
_manifest_audit.csv
clinical_label_dictionary.json
clinical_labels.csv
clinical_labels_complete4visits.csv
mri_nact_feature_dictionary.json
mri_nact_features_complete4visits_wide.csv
mri_nact_features_long.csv
mri_nact_features_wide.csv
mri_nact_features_with_clinical_labels.csv
```

Example patient layout:

```text
<ISPY2_PREPROCESSED_ROOT>/ISPY2-159352/
  manifest.json
  visit_summary.csv
  T0/
    ISPY2-159352_T0_original_DCE.nii
    ISPY2-159352_T0_original_DCE.json
    ISPY2-159352_T0_analysis_mask_raw.nii
    ISPY2-159352_T0_analysis_mask_raw.json
    ISPY2-159352_T0_ftv_mask.nii
  T1/
  T2/
  T3/
```

Each patient manifest records the raw series path, conversion output, DCE shape,
number of DCE time frames, mask values, bounding box, and any DCE layout
adjustment.

## Preprocessed I-SPY1 Root

```text
<ISPY1_PREPROCESSED_ROOT>
```

Top-level files:

```text
_ispy1_preprocess_summary.csv
clinical_labels_complete4visits.csv
```

Example patient layout:

```text
<ISPY1_PREPROCESSED_ROOT>/ISPY1_1156/
  manifest.json
  T0/
    ISPY1_1156_T0_original_DCE.nii
    ISPY1_1156_T0_original_DCE.json
  T1/
  T2/
  T3/
```

## DCE Timing Audit Outputs

The clean runner writes the DCE timing audit under:

```text
<ISPY2_PREPROCESSED_ROOT>/_audits/dce_dicom_timing_audit.csv
<ISPY2_PREPROCESSED_ROOT>/_audits/dce_dicom_timing_audit_summary.json
```

These files compare the old `frame 1 / last frame` policy with BreastDCEDL
metadata `post_early / post_late` indices.

## Clean Modeling Caches

The standalone default configuration creates:

```text
<ISPY2_PREPROCESSED_ROOT>/_corejepa_clean_dce8/<patient_id>.npz
<ISPY2_PREPROCESSED_ROOT>/corejepa_response_features.npz
```

Each patient tensor cache contains image `[4,8,32,96,96]`, geometry `[4,9]`,
phase indices, channel names, and ROI provenance. The response cache contains
raw pCR-free descriptors `[N,4,106]`. Training-split transforms are stored in
the model checkpoint, not baked into the raw response cache.

## Legacy Development Caches

Important derived caches under `<ISPY2_PREPROCESSED_ROOT>`:

```text
_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96
_mixed_ispy1_train_cache_dce8_meta_allpost_axiscanonv1_autoroi_z32_y96_x96
_mixed_supervised_cnn_cache_dce8_meta_allpost_z32_y96_x96
_mixed_janickova_inspired_cache_dce8_meta_allpost_z32_y96_x96
_fullcurve_phase8_t0crop_z32_y96_x96
_breastdcedl_mincrop_t0_raw224_v1
```

These names are retained only to identify historical development artifacts. The
clean package does not import code from them and uses the two caches above.
