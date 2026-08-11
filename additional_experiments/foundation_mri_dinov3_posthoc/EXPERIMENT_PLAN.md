# DINOv3 DCE-MRI post-hoc sensitivity baseline

This sibling experiment was requested after publication of the locked
`foundation_mri_baselines` results. It does not alter that experiment, its
candidate set, or its conclusions.

The only new encoder is the official Meta DINOv3 ViT-B/16 LVD-1689M model.
Its use is explicitly exploratory because the custom license requires local
institutional compliance and the non-enumerable web pretraining corpus does
not permit a patient-level I-SPY exclusion proof. The checkpoint is consumed
from an existing local cache in strict offline mode and is never redistributed.

Before any DINOv3 outcome is loaded, this experiment freezes:

- the official model revision and every local model artifact digest;
- the unchanged DCE input/channel/spatial/timing adapter;
- the 808-patient five-fold split and complete-case population;
- the linear/Ridge probe implementation and grids inherited byte-for-byte
  from the published baseline experiment;
- every reported candidate and matched comparison, with no best-cell filter.

Primary outputs remain frozen-encoder pCR probes. HR/HER2/subtype and
FTV/delta-FTV probes are secondary representation diagnostics. All confidence
intervals are descriptive paired outer-fold OOF patient bootstraps.

