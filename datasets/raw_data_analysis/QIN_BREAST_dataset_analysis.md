# QIN-BREAST Dataset Analysis for World Model Training

## Overview

This document summarizes a local inspection of the `QIN-BREAST_2015-09-04` dataset and evaluates whether it is suitable for training a medical world model, especially with a JEPA / LeJEPA-style objective.

## Dataset Structure

The dataset is organized in a standard DICOM hierarchy:

- `Patient`
- `Study`
- `Series`
- `DICOM files`

In this local copy, the dataset contains:

- **68 patients**
- **216 studies**
- **489 image series**
- **~93,117 DICOM files**

The files are stored under:

- `qin_breast/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`

There is also a metadata index file:

- `metadata/metadata.csv`

## What a DICOM Series Means

In this dataset, a **series** does **not** mean a longitudinal timeline such as "baseline MRI" and "follow-up MRI."

A DICOM **series** usually means one imaging acquisition within a study, for example:

- one MRI sequence
- one CT acquisition
- one PET acquisition

So the hierarchy should be interpreted as:

- **Patient**: the subject
- **Study**: one imaging exam or visit
- **Series**: one acquisition or scan sequence within that exam
- **Image files**: the individual DICOM slices or frames

## Input Modality

The dataset is **multimodal medical imaging data** in DICOM format.

From the DICOM headers sampled locally, the series-level modality mix is:

- **MR**: 291 series
- **CT**: 99 series
- **PT**: 99 series

This means the dataset is **not MRI-only**. It includes:

- **MRI**
- **CT**
- **PET** (stored as `PT` in DICOM)

Some series descriptions and protocol names observed in the files include:

- `CTAC`
- `PET AC 3DWB`
- `PRONE BREAST`
- `7.3 PRONE BREAST`

So a concise description of the input is:

> Multimodal breast imaging in DICOM format, containing MR, CT, and PT series.

## Ground Truth

No explicit supervised ground-truth targets were found in this local export.

Specifically, I did **not** find:

- segmentation masks
- lesion contours
- voxelwise annotations
- diagnosis labels
- pathology labels
- treatment outcome labels
- explicit prediction targets

The `metadata.csv` file appears to function mainly as an index of stored studies and series. In this copy, many potentially useful descriptive fields are empty, including fields related to:

- modality in the CSV table
- series description in the CSV table
- protocol name in the CSV table
- study date in the CSV table
- longitudinal temporal event type

Therefore:

> This dataset copy appears to contain raw imaging data only, with no explicit ground-truth file packaged alongside it.

## Labels Available

The dataset does contain **metadata-derived labels**, but these are mostly technical or organizational rather than clinical.

Examples include:

- `PatientID`
- `StudyInstanceUID`
- `SeriesInstanceUID`
- DICOM `Modality`
- protocol or series naming strings when present in the DICOM headers

These are useful for:

- grouping by patient
- grouping by visit / study
- grouping by imaging sequence
- separating MR, CT, and PT subsets

However, these are **not** disease labels or task labels in the usual supervised-learning sense.

## Are Time Points Included?

Possibly yes, but not clearly labeled.

The dataset includes **multiple studies for many patients**, which suggests that some patients may have been scanned at more than one visit or timepoint.

From the local inspection:

- 68 patients total
- 216 studies total
- many patients have more than one study

This means **longitudinal data may be present**.

However, an important limitation is:

- the local metadata export does **not clearly indicate baseline vs follow-up**
- explicit temporal fields appear mostly empty in this copy
- study ordering may need to be reconstructed from DICOM headers rather than from the provided CSV alone

So the correct interpretation is:

> Multiple time points may exist for some patients, but they are not cleanly exposed as labeled longitudinal events in this export.

## Is This Dataset Good for LeJEPA or a Medical World Model?

### Short answer

**Partly yes**, depending on what is meant by "world model."

### Good fit for self-supervised representation learning

This dataset is a **good candidate for JEPA-style or LeJEPA-style self-supervised pretraining** because it provides:

- a large amount of unlabeled medical imaging data
- repeated anatomical structure
- many slices per series
- multiple modalities
- natural grouping into patient / study / series

This is useful if the goal is to learn:

- strong medical image embeddings
- latent representations for downstream tasks
- spatially structured representations without manual labels

### Limitations for a true longitudinal world model

This dataset is a **weaker fit** for a full longitudinal medical world model because:

- a DICOM series is not the same as a disease trajectory
- explicit temporal ordering is weak in the local export
- follow-up labels are not clearly available
- no treatment or intervention actions are provided
- no explicit state-transition targets are provided

That means it is not ideal if the intended world model is:

- patient progression over time
- treatment response modeling
- disease evolution across clearly labeled visits

### Better interpretation for world modeling

If you use this dataset for a JEPA-style world model, the safest interpretation of "world" is:

- **anatomical structure within a scan**
- **local spatial continuity across slices**
- **cross-view or cross-modality consistency within a study**

Rather than:

- disease progression across visits

## Recommended Ways to Use This Dataset with LeJEPA

This dataset is most suitable for the following training formulations:

### 1. 2D slice-based JEPA

Train on individual slices and predict masked or withheld regions in latent space.

Useful when:

- simplicity matters
- you want a strong baseline
- you want to mix MR, CT, and PT with minimal engineering

### 2. 2.5D context prediction

Use a center slice with neighboring slices as context.

Useful when:

- you want limited volumetric structure
- full 3D training is too expensive

### 3. 3D volume or block JEPA

Treat each series as a 3D volume and predict missing subvolumes or withheld slices from context.

This is likely the most natural "world model" interpretation for this dataset.

Useful for learning:

- spatial continuity
- anatomical structure
- latent volumetric priors

### 4. Cross-modality learning within a study

Use one modality as context and another as target when studies contain paired MR / CT / PT information.

Useful if:

- patient studies are sufficiently aligned
- you want modality-invariant or modality-bridging latents

## Final Recommendation

This dataset is:

- **good for LeJEPA-style self-supervised medical representation learning**
- **good for spatial or volumetric world modeling inside scans**
- **not ideal as a primary dataset for longitudinal disease world modeling**

So the practical conclusion is:

> Use this dataset if your medical world model is meant to learn anatomical or scan-structure dynamics in latent space.
>
> Do not rely on it alone if your goal is a patient-level longitudinal progression model across clearly defined timepoints.

## Suggested One-Sentence Summary

The QIN-BREAST dataset is a multimodal DICOM breast imaging collection with MR, CT, and PT series that is well suited for self-supervised JEPA-style representation learning and spatial world modeling within scans, but it does not provide clean explicit ground truth or clearly labeled longitudinal follow-up structure for full patient-dynamics world modeling.
