# I-SPY2 Preprocessing

This folder contains the first-pass preprocessing code for the I-SPY2 DICOM data.

The initial target is one patient at a time. The script discovers the four MRI visits
(`T0`-`T3`), finds the VOLSER `original_DCE` and `Analysis_Mask` series in each visit,
converts them with `dcm2niix`, and writes a manifest with image shape and mask-derived
FTV bounding boxes.

Important mask convention:

- I-SPY2 analysis masks are inverse masks from FTV processing.
- Voxels with value `0` are included in the measured FTV.
- Non-zero values are bit-encoded exclusion reasons.

For the current DICOM SEG files, embedded metadata commonly reports these bits:

- `1`: PE threshold
- `2`: other SER / MNC-like filter
- `16`: automatic background threshold
- `32`: manual VOI

The project therefore uses `analysis_mask == 0` as the first-pass tumor/FTV mask.

## DCE handling

The preprocessing keeps the full 4D `original_DCE` volume for each visit instead of
selecting only BreastDCEDL-style `pre/post_early/post_late` phases. Phase selection can
therefore happen later in the dataset loader.

Some GE DCE series are converted by `dcm2niix` as multiple z-slabs in the time dimension
for example `z/2 x time*2`. When the analysis mask z-size and the DICOM acquisition-time
count make this unambiguous, the script writes an aligned DCE NIfTI and records the
operation in `dce_layout_adjustment`.

If `dcm2niix` emits multiple NIfTI files for one DCE series, the script selects the unique
largest output, which handles occasional extra single-volume echo outputs.

## Example

```bash
python ispy_jepa_tmi_clean/data_processing/preprocessing/preprocess_one_patient.py \
  --patient-id <PATIENT_ID> \
  --raw-root <ISPY2_RAW_ROOT> \
  --output-root <ISPY2_PREPROCESSED_ROOT>
```

Default input root:

```text
${ISPY2_RAW_ROOT}
```

Default output root:

```text
ispy_jepa_tmi_clean/data_processing/data/Preprocessed/I-SPY2
```

Full-data output used for this project:

```text
${ISPY2_PREPROCESSED_ROOT}
```

See `../config/paths.example.env` for the data-root variables expected by the
clean version of these scripts. Command-line arguments such as
`--raw-root`, `--output-root`, and `--dcm2niix` always override the environment
fallbacks.

Useful project-level files:

- `_batch_summary.csv`: runtime batch log.
- `_manifest_audit.csv`: manifest-level complete/partial audit after repair passes.
- `_logs/<patient>.log`: per-patient conversion log.

Clinical label files extracted from `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx`:

- `clinical_labels.csv`: all 985 clinical rows, aligned to preprocessed `patient_id`.
- `clinical_labels_complete4visits.csv`: the 808 patients with complete `T0`-`T3` imaging.
- `clinical_label_dictionary.json`: label meanings, missing counts, and value counts.

Main label columns are `label_pcr`, `label_hr`, `label_her2`, `label_mp`,
`age_at_screening`, `arm`, `hr_her2_subtype`, `race_simple`,
`menopausal_status_simple`, and `ethnicity`.

MRI NACT feature files extracted from `Multi-feature-MRI-NACT-Data.xlsx`:

- `mri_nact_features_wide.csv`: 384 patients with one row per patient.
- `mri_nact_features_long.csv`: 1536 rows, one row per patient per visit.
- `mri_nact_features_complete4visits_wide.csv`: 375 feature patients with complete local `T0`-`T3` imaging.
- `mri_nact_features_with_clinical_labels.csv`: MRI features merged with clinical labels.
- `mri_nact_feature_dictionary.json`: source-column mapping and feature summary statistics.

The wide feature table maps source `V10/V20/V30/V40` columns to local `T0/T1/T2/T3`
columns such as `tumor_volume_blu_t0`, `sphericity_t0`, `ld_t0`, and
`bpe_5slice_mean_t0`. Percent-change columns are kept as `*_pch_t0_t1`,
`*_pch_t0_t2`, and `*_pch_t0_t3`.
