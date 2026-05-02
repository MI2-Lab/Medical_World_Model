# HCC-TACE Dataset Analysis for World Model Training

## Overview

This document summarizes a local inspection of the `HCC-TACE-Seg_v1_202201` dataset and evaluates whether it is suitable for training a medical world model, especially with a JEPA / LeJEPA-style objective.

## Dataset Structure

The dataset is organized in a standard DICOM hierarchy:

- `Patient`
- `Study`
- `Series`
- `DICOM files`

In this local copy, the dataset contains:

- **105 patients**
- **211 studies**
- **677 series listed in metadata**
- **~39,371 DICOM files**

The files are stored under:

- `hcc_tace_seg/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`

There is also a metadata index file:

- `metadata/metadata.csv`

## What a DICOM Series Means

As with most DICOM datasets, a **series** does **not** mean a longitudinal sequence like "before treatment" and "after treatment."

A DICOM **series** means one imaging acquisition or one derived object within a study, for example:

- one CT acquisition
- one reconstruction
- one segmentation object

So the hierarchy should be interpreted as:

- **Patient**: the subject
- **Study**: one imaging exam or visit
- **Series**: one acquisition or derived object within that exam
- **Image files**: the individual DICOM slices or segmentation objects

## Input Modality

The main imaging input is **CT**.

From direct DICOM header inspection, the series-level modality mix is:

- **CT**: 444 series
- **SEG**: 76 series

The SOP classes confirm this:

- `1.2.840.10008.5.1.4.1.1.2` = **CT Image Storage**
- `1.2.840.10008.5.1.4.1.1.66.4` = **DICOM Segmentation Storage**

So the dataset is best described as:

> A liver CT dataset with accompanying DICOM segmentation objects.

Common CT series descriptions observed locally include:

- `PRE LIVER`
- `Recon 2: PRE LIVER`
- `Recon 3: LIVER 2 PHASE (C/A/P)`
- `Recon 2: LIVER 3 PHASE (AP)`
- `2.5 STANDARD`
- `C-A-P`

This suggests that the CT data includes multiple liver-focused acquisition/reconstruction phases rather than a single uniform series per visit.

## Ground Truth

Unlike QIN-BREAST, this dataset **does include explicit ground truth**.

The ground truth is provided as:

- **DICOM SEG objects** (`Modality = SEG`)

These are segmentation labels stored in DICOM format rather than as separate NIfTI or PNG masks.

From the local inspection:

- **76 segmentation series**
- **75 patients with at least one segmentation series**

This means the dataset has **real supervised labels for a subset of patients / studies**, not just metadata.

## Labels Available

The dataset contains several kinds of labels or supervision:

### 1. Image modality / technical labels

- `PatientID`
- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Modality` (`CT`, `SEG`)
- series descriptions such as liver phase/reconstruction names

These are useful for grouping and filtering the data.

### 2. Segmentation labels

The most important label type is:

- **voxelwise segmentation ground truth**, stored as DICOM SEG

This is the primary supervised signal in the dataset.

### 3. Implicit temporal / treatment structure

Because the dataset is named **HCC-TACE**, it is clearly related to hepatocellular carcinoma and TACE treatment context.

From the local structure:

- **104 patients have 2 studies**
- **1 patient has 3 studies**

This strongly suggests that the dataset is organized around **multiple visits / timepoints per patient**, likely including imaging before and after treatment-related events.

However, the local metadata CSV does not cleanly provide human-readable treatment phase labels such as:

- pre-TACE
- post-TACE
- baseline
- follow-up

So there is temporal structure, but some interpretation may still need to be reconstructed from dates and study relationships.

## Are Time Points Included?

Yes, much more clearly than in QIN-BREAST.

This dataset appears to include **longitudinal imaging**:

- **105 patients**
- **211 studies**
- almost all patients have **2 studies**

Direct DICOM header inspection also showed real study dates for example patients, with different dates across their studies.

Examples observed locally:

- `HCC_001`: `1999-11-30` and `2000-04-21`
- `HCC_006`: `1999-08-06` and `2000-01-22`
- `HCC_010`: `1998-02-21` and `1998-05-03`

So the correct interpretation is:

> This dataset does include patient-level time points, and the two-study structure is strong evidence of longitudinal follow-up.

That makes it much more promising for world-model style training than QIN-BREAST.

## Important Limitation About Labels

Even though segmentation labels are present, they are **not available for every patient**.

From the local inspection:

- **105 total patients**
- **75 patients with SEG labels**

So:

- the dataset is **partially labeled**
- some patients appear to have CT studies without matching segmentation objects

This matters for supervised training and for any evaluation pipeline that assumes full annotation coverage.

## Is This Dataset Good for LeJEPA or a Medical World Model?

### Short answer

**Yes, more than QIN-BREAST**, especially if your world model is based on longitudinal CT structure and lesion evolution.

### Why it is promising

This dataset has several properties that are valuable for world-model training:

- repeated **patient-specific time points**
- consistent **CT modality**
- explicit **segmentation supervision**
- disease-focused liver imaging
- a likely treatment-related longitudinal setting

Compared with QIN-BREAST, this dataset is much better aligned with the idea of:

- current state -> future state
- lesion appearance change over time
- treatment-related anatomical change

### Why it is still not perfect

There are still some important limitations:

- the CSV metadata is sparse and does not clearly name pre/post treatment stages
- segmentation is not available for every patient
- there are no explicit treatment-action fields packaged in this local export
- some series are reconstructions or processed derivatives, so careful preprocessing is needed

So it is better than a purely static imaging dataset, but it is not a complete structured decision-making dataset with explicit actions and outcomes.

## Recommended Ways to Use This Dataset with LeJEPA

### 1. Longitudinal CT latent prediction

Use one study as context and the later study as target for the same patient.

This is the most natural world-model formulation for this dataset.

Useful for learning:

- patient-specific temporal progression
- post-treatment appearance changes
- lesion evolution in latent space

### 2. 3D intra-study JEPA

Treat each CT series as a 3D volume and predict missing subvolumes or withheld slices.

Useful for learning:

- anatomical structure
- local volumetric consistency
- liver lesion context

### 3. Segmentation-aware representation learning

Use the DICOM SEG objects as supervised or semi-supervised guidance.

Useful for:

- lesion-focused latent learning
- evaluation of learned features
- combining self-supervised and supervised objectives

### 4. Longitudinal + segmentation setup

For patients with segmentations, use:

- earlier CT as input
- later CT as future state
- segmentation as an auxiliary target or region-focused supervision

This is likely the strongest medical-world-model setup available in this dataset.

## Final Recommendation

This dataset is:

- **good for LeJEPA-style medical representation learning**
- **good for longitudinal CT world modeling**
- **good for segmentation-aware medical latent learning**
- **much better than QIN-BREAST for patient-level temporal modeling**

However:

- it still needs careful study matching and preprocessing
- label coverage is incomplete
- explicit treatment/action annotations are not clearly packaged in this copy

So the practical conclusion is:

> HCC-TACE-Seg is a strong candidate for a medical world model if your goal is to model longitudinal liver CT changes and lesion structure, especially when combining JEPA-style latent prediction with segmentation supervision.

## Suggested One-Sentence Summary

The HCC-TACE-Seg dataset is a longitudinal liver CT dataset with DICOM segmentation labels for a substantial subset of patients, making it much better suited than QIN-BREAST for JEPA-style medical world modeling of temporal disease and treatment-related change.
