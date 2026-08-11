# Compact MRI–Clinical Fusion Audit

## Question and frozen scope

This diagnostic audit asks whether Goal 2's negative incremental pCR result was
caused by high-dimensional linear fusion rather than missing phenotype
information. It changes only the downstream MRI representation and fusion
strategy. The MRI encoder, JEPA, LOCAL architecture, checkpoints, pCR labels,
patient populations, outer folds, clinical/FTV contracts, and prediction timing
are frozen.

The experiment is based on Goal 2 commit `064e059` and runs on branch
`feature/compact-mri-clinical-fusion-audit`. It reads but never writes
`mri_clinical_complementarity_audit/` or
`local_global_response_state_pilot/`.

## Frozen inputs

- `full_808`: secondary `C` versus `C+M` estimand and profile probes.
- `ftv_complete_375`: primary fully matched C/F/M estimand.
- Outer folds: the exact five-fold seed-2026 patient manifest from Goal 2.
- MRI cells: seed bases 2026 and 3026, arms LOCAL0 and LOCAL3, folds 0–4.
- MRI tensors: frozen online pre-projector response states `[N,4,192]`.
- Clinical `C`: Goal 2 `C2_full_with_treatment`.
- FTV `F`: `log1p(FTV_T0 ... FTV_Tk)` at timing `Tk`.
- Classifier: train-standardized L2 logistic regression, Goal 2 C grid,
  validation-AUROC selection, smaller-C tie-break.

The source config, timing contract, final report, and clinical inventory are
SHA-256 pinned in `configs/audit.json`.

## Timing and compact representation contract

Goal 2 concatenated every observed 192-D visit state. Therefore its raw MRI
prefix is 192, 384, 576, or 768 dimensions at T0, T1, T2, or T3. The shorthand
`M192` means the uncompressed sequence of 192-D visit states; tables always
publish the actual prefix dimension.

For this audit, `Mk` means:

1. construct the same timing-safe raw prefix as Goal 2;
2. fit one centered PCA on outer-train patients only; and
3. transform that complete prefix to **k total dimensions**, where
   `k in {8,16,32,64}`.

One 64-component PCA is fitted per population×seed×arm×outer-fold×timing;
candidate `Mk` matrices use its leading k components. This nesting ensures all
candidate dimensions share the same fitted basis. PCA never sees validation or
test patients. The downstream logistic C and k are chosen on outer validation
AUROC, with smaller k and then smaller C as deterministic tie-breaks. Every
prespecified dimension is transformed/evaluated on test for the dimension
sensitivity plots, but the headline `Mk` row is locked from validation before
test metrics are read.

Explained-variance ratios, fitted-transform hashes, and selected k by fold and
model family are saved. Patient-derived PCA parameters remain private under
`features/`; aggregate explained variance and selection ledgers are public in
`metrics/`.

## Random-projection control

Gaussian random projections to 16 and 32 total dimensions are generated from
the fixed configured seed. Their matrices depend only on raw feature dimension,
timing, and the seed; they never read patients or labels. Both prespecified
dimensions are reported as secondary sensitivities and neither is selected by
test performance.

## Model families

At each timing:

- both populations: `C`, raw/compact/RP `M`, and `C+M`;
- `ftv_complete_375`: `F`, `C+F`, and raw/compact/RP `C+F+M`;
- `ftv_complete_375`: compact `M_residual` and
  `C+F+M_residual`, after an outer-train-only linear FTV-prefix→Mk fit;
- both populations: `LateFusion(C,Mk)`;
- `ftv_complete_375`: `LateFusion(C+F,Mk)`.

Residualization is ordered PCA first, FTV regression second. Both transforms
are fitted only on outer train and frozen for validation/test.

## Strict late fusion

For each candidate k, five deterministic stratified inner folds partition only
the outer-train patients. Every meta-training logit is predicted by a base model
whose clinical encoder, scaler, PCA, and coefficients were fitted without that
patient. The chosen outer-fold hyperparameters are fixed during this inner-OOF
generation. The two meta features are clipped logits `[l_clinical, l_MRI]`.

The two-dimensional L2 logistic meta-model is fitted on strict inner-OOF
outer-train logits. Final (k, base C, meta C) selection uses only outer-validation
AUROC with the same smaller-k/smaller-C tie rules, then evaluates outer test
once. In-sample outer-train logits are forbidden and a private OOF ledger plus
public coverage diagnostics are saved.

## Profile sensitivity

Raw prefixes, PCA-16, and PCA-32 are probed for HR, HER2, and observed four-class
HR/HER2 subtype at T0–T3. PCA remains pCR- and profile-label agnostic and is fit
on outer train only. Binary probes use balanced class weights; subtype uses
balanced multiclass L2 logistic regression.

## Effects and uncertainty

Primary comparisons are:

1. `C+Mk - C`;
2. `C+F+Mk - (C+F)`;
3. `LateFusion(C,Mk) - C`;
4. `LateFusion(C+F,Mk) - (C+F)`.

Raw-versus-compact and residual-beyond-FTV effects are also reported. For every
population×seed×arm×timing cell, 2,000 paired patient bootstrap draws resample
within outer-fold strata. AUROC/AUPRC effects are comparison minus reference.
`delta_brier` is comparison minus reference (negative is favorable), while the
companion `brier_improvement` is reference minus comparison (positive is
favorable). MRI seeds/arms are sensitivity cells, never independent patients.

## Pre-registered interpretation

- **A. Complementary signal revealed by compact fusion:** at least one T0/T1/T2
  timing is directionally positive across both seed bases, shows a clear paired
  CI improvement trend, and compact beats raw.
- **B. Late fusion only supported:** concatenation fails but strict OOF late
  fusion stably improves the clinical reference.
- **C. Dimensionality not the bottleneck:** PCA, random projection, and late
  fusion all fail to improve the clinical reference.
- **D. Mixed / unstable:** findings depend materially on seed, arm, timing, or
  model family and do not satisfy A–C cleanly.

T3 is a late/pre-surgery response assessment and cannot alone satisfy the
early/mid criterion. The two arms are reported separately within each seed; no
cross-population absolute score is interpreted as a paired effect.
