# HCC-TACE-Seg Dataset

## Basic Info
- **Disease**: Hepatocellular Carcinoma (HCC)
- **Patients**: 105
- **Modality**: CT (pre- and post-TACE)
- **Timepoints**: Pre-TACE, Post-TACE (1-12 weeks after procedure)
- **Treatment**: Transarterial Chemoembolization (TACE)
- **Outcome**: Time-to-progression, Overall survival

## Why it matters for World Model
- **State**: Pre-treatment CT with tumor segmentation
- **Action**: TACE procedure
- **Next State**: Post-treatment CT + tumor changes
- **Outcome**: Survival analysis, Tumor progression

## Strengths
- Detailed radiomic features for tumor prediction.
- Clear clinical outcome measures: survival and progression.

## Weaknesses
- Data is retrospective.
- Limited to only pre- and post-TACE imaging (no continuous follow-up).

## Preprocessing Needs
- CT image normalization.
- Tumor segmentation preprocessing.
- Radiomics feature extraction.

## Current Decision
- **Status**: Selected
- **Use Case**: Survival prediction based on pre-treatment CT.

## Notes
- This dataset is useful for **predicting survival** and modeling tumor response to TACE.
