# Frozen experiment protocol V2

## Question

Without using pCR, clinical variables, FTV, radiomics, ROI masks, or geometry as
inference inputs, does a frozen DINOv3 backbone followed by an MRI-domain
adapter and direct residual-radiomics grounding learn image representation that
adds information beyond clinical+FTV?

## Outcome-blind representation phase

- Cohort: frozen C1B-H technical population (947 patients), seed-2026 five-fold
  membership; each fold adds 139 I-SPY1 patients to training only.
- DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`, frozen permanently.
- Input: central 64 mm C1B LOCAL cube, 32 axial slices, seven DCE channels.
- Cache: per channel and slice concatenate CLS, patch mean, and patch standard
  deviation after excluding register tokens, yielding 2304 values.
- Arms: D1 JEPA+SIGReg; D2 D1+0.25 FTV; D3 D2+0.10 radiomics-PC. All include the
  same 16-dimensional radiomics head.
- Matrix: seeds 2026/3026/4026/5026/6026 x folds 0..4 x D1/D2/D3 = 75 cells.
- Checkpoint selection reads validation representation/grounding losses only.

## Radiomics target feasibility revision

The observable ROI is resampled FTV support intersected with the central LOCAL
cube and valid-source mask. Original-image first-order, GLCM, GLRLM, GLSZM,
GLDM, and NGTDM features are extracted separately for all seven C1B channels
using PyRadiomics 3.1.0, force2D axial, bin width 0.25, no normalization or
internal resampling. Shape, wavelet, and LoG are disabled. Original, in-plane
erosion, and in-plane dilation are all bounded by valid-source support. Target
ROIs remain at 64 voxels and three slices; T2 coverage is prospectively revised
to 85%. Outward stability is measured on all valid visits, while symmetric
stability is measured only where erosion remains extractable. Fold-train-only
stability filtering, FTV/volume/visit residualization, and PCA16 use T0--T2;
T3 is descriptive and its radiomics loss mask is always false.

## Mechanism and evaluation locks

The pCR evaluator refuses to import or open an outcome table before the matrix
completion and frozen-state lock verify all expected files and SHA-256 hashes.
An outcome-blind mechanism audit must then show D3 residual-PC macro Spearman
at least 0.10, D3-D2 gain at least 0.05 in at least four seeds, no material FTV
loss, and complete optimization safety before `MECHANISM_LOCK.json` is written.
After unlock, the 375-patient primary analysis uses timing-safe clinical+FTV
offset fusion with five fixed 32-D Gaussian projections and inner-OOF logits.
All success thresholds and decision labels are encoded in `configs/protocol.json`.
