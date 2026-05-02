# QIN-BREAST Dataset

## Basic Info
- **Disease**: Breast Cancer
- **Patients**: ~30-40
- **Modality**: PET/CT, MRI (DWI, DCE, T1 mapping)
- **Timepoints**: Pre-treatment, First cycle of treatment, Post-treatment (prior to surgery)
- **Treatment**: Neoadjuvant therapy (Chemotherapy)
- **Outcome**: Early treatment response (imaging-based)

## Why it matters for World Model
- **State**: Pre-treatment PET/CT, MRI
- **Action**: Neoadjuvant therapy (chemotherapy)
- **Next State**: Imaging post-treatment (tumor changes)
- **Outcome**: Tumor response (measured by tumor shrinkage or progression)

## Strengths
- Longitudinal imaging data with multiple time points.
- Multi-modal data (PET/CT + MRI) for a comprehensive view of tumor changes.

## Weaknesses
- Limited number of patients.
- No survival data or long-term follow-up (focus on early response).

## Preprocessing Needs
- PET/CT and MRI data alignment.
- Tumor segmentation and tracking.
- Dynamic imaging normalization.

## Current Decision
- **Status**: Selected
- **Use Case**: Early treatment response prediction.

## Notes
- Ideal for building models to predict **early treatment response** in breast cancer using imaging biomarkers.

