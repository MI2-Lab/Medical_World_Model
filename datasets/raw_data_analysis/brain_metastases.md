# Brain Metastases Longitudinal Dataset

## Basic Info
- **Disease**: Brain Metastases (from lung, breast, etc.)
- **Patients**: 40
- **Modality**: MRI, CT, Radiotherapy (RT) Plan
- **Timepoints**: Pre-treatment, Follow-up (6w, 3m, 6m, 9m, 12m)
- **Treatment**: Radiotherapy (Stereotactic Radiosurgery, WBRT)
- **Outcome**: Tumor segmentation (necrotic core, edema, enhancing tumor)

## Why it matters for World Model
- **State**: MRI scans + tumor segmentation
- **Action**: RT dose map
- **Next State**: Follow-up MRI + tumor mask
- **Outcome**: Tumor progression / shrinkage

## Strengths
- Longitudinal data with multiple follow-up time points.
- Detailed tumor segmentation (necrotic core, edema, enhancing tumor).
- Radiotherapy treatment details included.

## Weaknesses
- Small patient size (40 patients).
- Tumor segmentation is manual, which may introduce variability.

## Preprocessing Needs
- Image registration (MRI and CT).
- Tumor segmentation refinement.
- Normalization of dose map for model input.

## Current Decision
- **Status**: Selected
- **Use Case**: Dose-conditioned tumor response prediction.

## Notes
- This dataset is ideal for building a **treatment response prediction model*
