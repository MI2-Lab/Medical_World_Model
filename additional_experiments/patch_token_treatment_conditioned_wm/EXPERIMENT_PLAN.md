# Patch-Token Treatment-Conditioned World Model Pilot

## Scientific question and boundary

This pilot asks whether the confirmed 192-D, fixed-LOCAL pooled response state
discards treatment-response or phenotype information that remains in the final
spatial MRI latent field.  The only new scientific mechanism is outcome-free,
assigned-treatment-conditioned prediction of future spatial tokens inside the
same fixed 64-mm C1B-H LOCAL support.

World-model training is pCR-free.  pCR is loaded only after every world-model
checkpoint and representation transform has been frozen.  Assigned-regimen
conditioning is not a causal treatment effect; all claims use the wording
**assigned-treatment-conditioned longitudinal latent modeling**.

## Required prior evidence

The following completed experiments were read before this plan and the formal
configuration were written.

1. `local_response_state_multiseed_confirmation` established
   `LOCAL_MULTISEED_CONFIRMED`.  Across five training seeds, LOCAL0-GAP0 had
   mean static and delta-FTV Spearman gains of +0.208 and +0.174; LOCAL3-LOCAL0
   added +0.030 and +0.032.  LOCAL3 is therefore the locked base.
2. `local_global_response_state_pilot` showed robust LOCAL > GAP, whereas a
   LOCAL+GLOBAL branch did not improve LOCAL.  It fixed C1B-H, the 64-mm
   fractional LOCAL mean, 192-D response projection, FTV weight 0.25,
   SIGReg, EMA, optimizer, folds, and logical-batch contract.
3. `c1b_spatial_pooling_bottleneck_audit` classified the failure as a pooling
   bottleneck.  The final map is dynamically derived as 128x14x22x20 for the
   locked input.  LOCAL recovered most of the GAP FTV deficit; FTV grounding
   remained stable but modest and predictions remained compressed.
4. `spatial_heterogeneity_phenotype_audit` found that Mean+Std did not recover
   robust HR/HER2/subtype/pCR or beyond-FTV information.  Its only Oracle
   support was a narrow mask-based PERI20 pCR result.
5. `mask_free_region_aware_audit` found that fixed crop-coordinate shells did
   not reproduce that Goal-5 Oracle signal.  The exact 64-mm support contains
   500 positive-overlap feature cells, 308 with fractional boundary weight;
   the effective weight sum is 316.0493815.
6. `mri_clinical_complementarity_audit` found no reliable incremental value
   from the current pooled LOCAL state beyond clinical or clinical+FTV
   baselines.  It also fixed the outer-fold train/validation/test probe and
   patient bootstrap protocol.

The joint inference is deliberately narrow: hand-crafted channel statistics
and coordinate shells did not recover phenotype complementarity, so this
pilot tests whether retaining spatial tokens and learning longitudinal
relations is the missing mechanism.  The prior work does not guarantee that
patch tokens will work.

## Frozen population and input contract

- Technical eligibility: the existing 947-patient C1B-H population: 808
  fold-assigned I-SPY2 patients plus 139 authorized external I-SPY1
  train-only patients.
- Exact seed-2026 patient fold manifest, five outer folds, no refill or patient
  movement.  Evaluation is I-SPY2 only; train-only patients never enter probes.
- C1B-H DCE7 float32 `[B,4,7,112,176,160]`, tensor order ZYX, spacing XYZ
  `[0.9,0.9,2.0]` mm, true RAS+, T0 anchoring, header-based longitudinal
  strategy, and unchanged visit definitions.
- T3 is always described as `late/pre-surgery`.
- The encoder remains four 3-D residual stages with widths 16/32/64/128 and
  strides 1/2/2/2.  No crop, encoder, lesion mask, segmentation attention,
  pixel reconstruction, or outcome-informed region is introduced.
- The full final map shape is derived from the frozen encoder geometry and
  validated against the runtime tensor; `14x22x20` is an expected value, not
  a hard-coded model assumption.

## Primary arms and training matrix

Training seeds are 2026 and 3026; effective per-fold seed is `seed_base+fold`.

- **A0 / LOCAL3 reference**: the already confirmed, immutable selected LOCAL3
  checkpoints and OOF features for the same two seeds and five folds.  Their
  source SHA-256 values are recorded in a private run manifest.  Reusing the
  confirmed reference avoids outcome-driven retraining and exactly reproduces
  the locked LOCAL3 behavior.
- **A1 / PATCH3**: ten new cells (2 seeds x 5 folds), with the same encoder,
  FTV grounding, optimizer family, EMA philosophy, logical batch 32, 12-epoch
  maximum, patience 4, and test-blind selection.

No A1 outcome metric is inspected until all ten selected checkpoints and their
world-model diagnostics are frozen.

## Spatial token contract

The audited fractional 64-mm LOCAL map is recomputed from the frozen geometry.
Every final feature cell with strictly positive overlap is retained, yielding
500 tokens under formal geometry.  Boundary cells remain tokens; their
fractional weights are used only when an exact LOCAL weighted mean is required.
Token JEPA loss is an unweighted mean across sampled cells.

Each raw 128-D cell is mapped by a learned `Linear(128,128)+LayerNorm(128)`.
Crop-centered physical feature-cell coordinates are computed in XYZ millimetres
from the actual grid, stride, spacing, and center offset.  They are normalized
by the 32-mm LOCAL half-width and passed through the locked physical positional
encoding.  Absolute scanner coordinates are not used.

These are overlapping-receptive-field tokens, not independent 8-voxel image
patches: the final theoretical receptive field is approximately
42.3x42.3x94.0 mm XYZ.

## Predictor and masking

The predictor uses the **condition-token** method, locked before outcome
evaluation.  FiLM is not an alternative arm.

- token width 128; four Transformer blocks; eight heads; feed-forward width
  512; dropout 0.1;
- source sequence: one transition-condition token plus all 500 current MRI
  tokens;
- query sequence: 250 learned mask queries carrying the future-cell physical
  position;
- the attention mask prevents condition/current context from reading query
  states; target MRI token values are never predictor inputs;
- each transition is predicted only from its current/preceding observed MRI
  state and baseline-known/nominal condition, never a future visit;
- target encoder and target token projection are stop-gradient EMA copies.

Exactly 50% of future positions (250/500) are sampled without replacement for
every patient and transition.  Formal masks are deterministically keyed by
effective seed, epoch, logical-batch position, patient identity hash, and
transition, so they are outcome-blind and stable under physical-microbatch
accumulation.  Validation/export masks use fixed deterministic keys.

The masked patch loss is normalized-token MSE, averaged across channels,
sampled cells, transitions, and patients with transition weights 2/1/0.5.
MRI pixels are never reconstructed.

## Transition condition

Clinical/treatment data enter only the predictor condition token.  They are
never concatenated to online or target MRI tokens and never enter the frozen
MRI readout directly.

- fixed 14-level assigned-arm vocabulary (13 I-SPY2 arms plus
  `ISPY1_NACT`), fixed before folds;
- HR, HER2, and MP-as-provided;
- age and an age-missing indicator are retained to honor the existing clinical
  condition contract; age mean/std are fitted on outer training plus authorized
  external train-only patients only;
- the prior seven nominal temporal bits: target T1/T2/T3 one-hot and observed
  prefix T0..T3;
- a scalar nominal adjacent-visit interval `delta_t=1.0`.

The inherited assets contain no measured scan-day interval.  Consequently
`delta_t` is explicitly unitless/nominal and constant for adjacent transitions;
it must not be reported as elapsed biological time.  This resolves the source
ambiguity before outcome evaluation rather than inventing dates.

Training clinical CSV reads use an exact pCR-free column allow-list.  pCR is not
loaded by the condition dataset, checkpoint selector, loss, token mask, PCA, or
world-model diagnostic.

## Objective and checkpoint selection

For A1:

`L = L_patch_JEPA + 0.09 L_SIGReg + 0.25 L_FTV`.

FTV uses the exact fractional weighted mean of the current raw 128-D LOCAL
cells, followed by the canonical `Linear(128,192)+LayerNorm(192)` response
projection and linear head.  FTV neither selects/masks tokens nor changes
attention.  The FTV transform is fitted on outer-train observable values only:
`log(FTV+epsilon)`, 1/99 winsorization, median/IQR.  Delta-FTV is not supervised.

Training uses AdamW (learning rate 5e-5, weight decay 1e-4), physical batch 4,
eight-way accumulation, logical batch 32, one exact logical SIGReg reduction,
one gradient clip at 5, one optimizer step, and one EMA update at momentum
0.996 per logical batch.  Maximum epochs are 12 and patience is 4.

A1 checkpoint selection uses validation patch loss among finite, non-collapsed
epochs; validation FTV is the secondary tie-break.  Test FTV, delta-FTV, pCR,
clinical complementarity, and test token diagnostics are forbidden selection
inputs.

## Frozen representation readouts

All world-model parameters are frozen before representation evaluation.  For
each seed/fold, the A1 online MRI tokens are exported for the 808 primary
patients; no condition is added to these tokens.

- `M_mean`: exact fractional-weighted mean of 128 projected LOCAL tokens.
- `M_pca64`: 64-component randomized PCA of the ordered, weighted flattened
  token field.  PCA centering/components are fitted on outer-train patients
  only, pooling their four visit rows; validation/test are transform-only.
- primary `M_patch`: `M_mean || M_pca64`, exactly 192 dimensions and
  attention-free.  No pCR-supervised attention or component selection exists.

The unweighted token mean is descriptive only.  The A0 feature is its confirmed
192-D online pre-projector LOCAL response state.  Longitudinal prefixes flatten
available per-visit 192-D states in chronological order.

## Endpoints and fold-safe evaluation

FTV and literal observed delta-FTV use fold-fitted ridge regression with alpha
grid `1e-4..1e3`, validation analysis-space MSE selection, smallest-alpha tie
break, and one untouched outer-test evaluation.  Metrics are Spearman, Pearson,
natural R2, RMSE, MAE, prediction/target variance ratio, and calibration slope.

pCR uses L2 liblinear logistic regression, C grid `1e-4..100`, validation AUROC
selection with smaller-C tie break, and reports AUROC, AUPRC, and Brier.  The
clinical baseline C is the prior primary `C2_full_with_treatment`: HR, HER2,
MP, age, race, menopausal status, ethnicity, and assigned arm, with all
imputation/encoding/scaling fitted on outer train only.  Causal FTV feature F is
the available `log1p(FTV)` prefix only.

Primary pCR comparisons are MRI only, `C+M` versus `C`, and `C+F+M` versus
`C+F`, at T0, T0-T1, and T0-T2.  T0-T3/late is descriptive.  Fold-specific OOF
predictions are pooled before scoring.  Folds and visits are never independent
replicates; training seed is the replicate.

Paired effects E1-E6 use identical patient sets.  Patient bootstrap samples
with replacement within outer-fold strata, with 2,000 draws and seed 260812.
No identifiers or patient-level predictions enter Git.

## Spatial dynamics diagnostics

For each future visit, token errors are summarized in three fixed,
outcome-blind coordinate bands inside the same LOCAL support: central
`max(|x|,|y|,|z|)<=16 mm`, inner LOCAL `16<max(...)<=24 mm`, and outer LOCAL
`max(...)>24 mm`.  Bands are descriptive; they do not change training or
readout and are not lesion/peritumoral regions.

The shuffled-time control cyclically remaps target visits within a patient from
`[T1,T2,T3]` to `[T2,T3,T1]`, while keeping source tokens, cell positions, and
mask keys fixed.  It is a diagnostic baseline, not an alternative trained arm.

## Gates and classification

- Gate A (`PATCH_DYNAMICS_VALID`): all ten A1 cells are finite; both seed-level
  target and prediction token SD are >=0.05; and both seeds have actual-time
  masked-token cosine at least 0.05 above cyclic shuffled-time cosine (or an
  equivalent >=5% normalized-MSE improvement, reported separately).
- Gate B: static FTV macro Spearman `A1-A0 > -0.03` in both seeds; delta macro
  Spearman `A1-A0 > -0.03` in both seeds (the operational definition of no
  systematic degradation).
- Gate C (`PATCH_STATE_ADDS_INFORMATION`): at least one registered static FTV,
  delta-FTV, or MRI-only pCR macro effect is positive in both seeds and its
  two-seed mean is >=+0.03.
- Gate D (`PATCH_STATE_COMPLEMENTARITY_SUPPORTED`): at T0-T1 or T0-T2,
  `C+F+M_patch - (C+F)` AUROC is strictly positive in both seeds.  The pooled
  patient-bootstrap 95% CI excluding zero is a preferred strengthening result,
  not an additional hard gate.

Classification precedence is: all gates ->
`PATCH_WORLD_MODEL_BREAKTHROUGH`; A+B with response gain but no meaningful
MRI-only pCR gain -> `RESPONSE_ONLY_GAIN`; A+B+C with MRI-only pCR gain but D
failure -> `PATCH_DYNAMICS_BUT_NO_COMPLEMENTARITY`; otherwise
`POOLED_LOCAL_REMAINS_SUFFICIENT`.  Missing formal cells produce
`INCOMPLETE_NO_SCIENTIFIC_CLASSIFICATION`, never an inferred result.

## Delivery and privacy

Tracked deliverables are aggregate metrics, figures, code/tests, the lock,
decision JSON, and a Chinese final report.  Raw MRI, identifiers, checkpoints,
private token features, condition rows, bootstrap draws, and patient-level
predictions are gitignored.  A final path-scope and privacy audit must pass
before the requested non-force push.

