# Spatial Heterogeneity / Phenotype Pooling Audit

## 1. Scientific question

This audit asks whether the frozen C1B `LOCAL` encoder feature map contains tumor
heterogeneity, morphology, or peritumoral phenotype information that is lost by
first-order mean pooling. The primary biological endpoints are HR, HER2, and the
four-class HR/HER2 subtype. pCR is a complementarity endpoint: it tests whether
the image statistics add response information beyond the locked clinical and FTV
contracts.

Stage A is a frozen-feature audit. It does not train or update an encoder. Stage B
is a single, conditional feasibility pilot and is authorized only if Gate A or
Gate C passes after Stage A has been completed.

## 2. Immutable upstream contract

- Arms: `LOCAL0`, `LOCAL3`.
- Seed bases: `2026`, `3026`.
- Outer folds: `0..4`, using the exact locked patient assignments.
- Checkpoint: the already selected, test-blind `selected.pt` for each cell.
- Encoder tensor: complete output of `model.encoder`, before spatial pooling and
  before `response_projection`.
- Formal input: float32 `[B,4,7,112,176,160]` C1B-H DCE7.
- Expected encoder tensor: float32 `[B*4,128,D,H,W]`. The spatial shape is derived
  from the actual output. The anticipated C1B shape `[14,22,20]` is not accepted
  unless observed at runtime.
- LOCAL support: the exact shared `c1b_local_weight_final` fractional sampling-cell
  overlap with the central 64 x 64 x 64 mm physical cube. It is hash-bound to the
  prior spatial audit and is never adapted by mask, FTV, label, or performance.
- P1 parity: applying the frozen checkpoint response projection to the extracted
  P1 mean must reproduce the immutable upstream LOCAL response-state asset within
  `rtol=1e-5, atol=1e-6` for every value.

The extraction streams the full pre-pooling map through deterministic statistics;
it does not persist multi-gigabyte raw maps. Shape, dtype, finite-value, checkpoint,
patient-order, split, and P1 parity evidence are persisted for every cell.
Before any cache-backed geometry or image read, a one-time preregistered preflight
content-hashes all 947 cache files in the already hash-bound C1B manifest: the
locked primary 808 plus the 139 external train-only patients required for canonical
Stage-B `train_all` fairness. Stage-A extraction and Oracle consumers select the
authenticated `primary` 808 subset; Stage B may train on both authenticated cohort
labels but still validates/tests only on primary patients. Later consumers recheck
every live size/mtime without repeating the one-time content hashing.

## 3. Preregistered mask-free pooling variants

Let `w(x)` be the fixed LOCAL fractional physical-overlap weight and `F_c(x)` the
raw channel value.

| ID | Definition | Dimension | Role |
|---|---|---:|---|
| P1 | weighted mean `sum(wF)/sum(w)` | 128 | current LOCAL reference |
| P2 | weighted population SD `sqrt(sum(w(F-mu)^2)/sum(w))` | 128 | variation-only diagnostic |
| P3 | `[P1,P2]` | 256 | primary phenotype candidate |
| P4 | weighted empirical inverse-CDF Q25/Q50/Q75 | 384 | secondary robust candidate |
| P5 | `[P1,P2,P4]` | 640 | diagnostic upper bound only |

Quantiles use the first sorted value whose cumulative positive LOCAL weight reaches
`q * sum(w)`; they are not interpolated and no other quantiles are tested. P5 uses
the separately locked strong-regularization grid and cannot directly authorize a
future architecture. No pooling feature contains lesion volume, mask geometry,
bounding box, FTV, spacing, acquisition FOV, coordinates, or an occupancy scalar.

## 4. Fold isolation and metrics

Every probe is fit separately inside an outer fold. Scaling, categorical encoding,
FTV residualization, hyperparameter selection, and binary decision-threshold
selection use outer train/validation only. The outer test partition is predicted
once. The five test partitions are pooled into one OOF metric per seed, arm, view,
target, and feature variant.

- Binary HR, HER2, and pCR: AUROC, AUPRC, validation-selected balanced accuracy.
- Four-class subtype: macro one-vs-rest AUROC, macro AUPRC, argmax balanced accuracy.
- pCR additionally: Brier score.
- Binary phenotype probes use balanced class weights; pCR preserves the Goal-2
  unweighted logistic contract.
- All variants use L2 logistic regression. P1-P4 use the general C grid. P5 and
  high-dimensional oracle concatenations use the strong grid capped at `C=0.1`.

Phenotype views are single-visit `T0`, `T1`, `T2`, and `T3`. Mask-free pCR views are
causal prefixes `T0`, `T0-T1`, `T0-T2`, and `T0-T3`; no view uses a future image.
Every prefix is concatenated in visit-chronological order, and within each visit
in registered statistic-component order then channel order. Thus the P1 prefix
dimensions are 128/256/384/512, P2 the same, P3 256/512/768/1024, P4
384/768/1152/1536, and P5 640/1280/1920/2560 for one through four visits.
Combined models concatenate `[clinical C, causal log1p FTV prefix, causal MRI
prefix]` in that block order.
MRI-only pCR is reported on both the full 808-patient cohort and the locked
FTV-complete 375-patient cohort. The latter is the matched primary comparison.

## 5. Beyond-FTV and residualization

The primary clinical vector `C` is the exact Goal-2 `C2_full_with_treatment`
contract: HR, HER2, MP, age, race, menopausal status, ethnicity, and treatment arm,
with train-only preprocessing. `F` is the causal `log1p` FTV prefix through the
current timing. On the same 375 patients, fit:

- `C`
- `C+F`
- `C+F+P1`
- `C+F+P3`
- `C+F+P4`

The primary contrasts are `C+F+P3` vs `C+F+P1` and `C+F+P3` vs `C+F`.

For P1 and P3 separately, fit an unregularized multi-output ordinary least-
squares map with intercept from the causal, chronological `log1p` FTV prefix to
the raw causal MRI prefix. A `StandardScaler` (outer-train mean and population
variance) is fit to FTV predictors only; MRI responses are not scaled. Fit on
outer train, apply the train-fitted prediction to train/validation/test, and
subtract it from the raw MRI prefix. Probe `P1_res`, `P3_res`,
`C+F+P1_res`, and `C+F+P3_res` for pCR.

## 6. Physical-space oracle localization

This section is diagnostic and is never a deployable model input. The private
sidecar contains all 808 locked primary patients, in locked fold-0 order, and all
four visit slots (3,232 slots). All 3,232 source-mask paths are inventoried, but a
mask is opened, content-hashed, and mapped only when that visit's authenticated
C1B cache has `support_available=true`. This visit-local authority yields 1,933
source-authorized visits: 808 at T0 and 375 at each of T1/T2/T3. The other 1,299
visit slots remain invalid without opening their masks; there is no all-four-visit
patient prefilter. On exactly the 1,500 visits marked by upstream
`c1b_oracle_valid`, reconstructed unconfined CORE occupancy must be bitwise equal
to the prior hash-bound oracle. The additional 433 authorized T0 visits are
authenticated by their cache-bound source-support hashes. Euclidean distances are
computed in millimetres with ZYX sampling `[2.0,0.9,0.9]`, never in voxel units.

- `Rcore`: lesion voxels.
- `Rperi10`: valid-source, non-lesion voxels with distance `(0,10]` mm.
- `Rperi20`: valid-source, non-lesion voxels with distance `(10,20]` mm.
- `Rlocal_rest`: valid-source voxels whose centers lie inside the fixed central
  64-mm cube and are not in the preceding three regions.

Each binary region is mapped to the final grid using the frozen theoretical-RF
fractional occupancy operation. Before confinement, reconstructed CORE occupancy
must be bitwise equal to the hash-bound upstream oracle. Every region occupancy
is then multiplied by the exact fixed LOCAL fractional sampling-cell weight, so
all oracle and fixed-P3 features use the identical 500-cell support and Gate C
cannot gain an expanded FOV through the large receptive field. Empty post-
confinement support has no fallback and is marked invalid; this post-LOCAL mapped
emptiness is the only region-invalidity rule on a source-authorized visit. Region comparisons use
the pair-specific complete-case intersection for exactly the compared features;
CORE_PERI requires CORE, PERI10, and PERI20, but never unrelated LOCAL_REST.
Each region contributes mean+SD (256-D). The preregistered oracle variants are
`CORE`, `PERI10`, `PERI20`, `LOCAL_REST`, and the strongly regularized concatenation
`CORE_PERI=[CORE,PERI10,PERI20]`. They are probed for HR, HER2, subtype, and pCR.
Fixed LOCAL P3 is re-evaluated on the identical oracle cohort for Gate C.

At timing T0 only the T0 mask and its current validity are used; at any later causal
prefix, only masks and validity through that timing are used. A future lesion mask
or future-visit availability is forbidden even for population selection in the
oracle diagnostic. Pair-specific sample sizes may therefore vary (T0 cohorts can
include up to 808 patients; later/prefix cohorts up to 375), while the exact Table 7
design still contains 640 metric rows. The de-identified representative is selected
only from the exact 375 patients with CORE valid at all four visits, using the stable
median total CORE input-voxel count in seed-2026/LOCAL3/fold-0.

## 7. Longitudinal heterogeneity

For each adjacent transition `T0->T1`, `T1->T2`, and `T2->T3`, probe pCR using:

- `DELTA_MEAN = mean_end - mean_start` (128-D),
- `DELTA_STD = std_end - std_start` (128-D),
- `P3_PLUS_DELTA = [P3_start, P3_end - P3_start]` (512-D).

These are deterministic frozen changes and receive no direct supervision.

## 8. Operational gates

All AUROC deltas below compare pooled OOF metrics at the same arm, view/timing,
endpoint, population, and seed.

- **Gate A — HETEROGENEITY_SIGNAL_SUPPORTED:** for at least one fixed arm and
  endpoint/view, P3-P1 AUROC is at least `+0.03` in both seeds for HER2 or subtype;
  or P3-P1 MRI-only pCR AUROC is at least `+0.03` in both seeds on the matched
  375-patient cohort.
- **Gate B — HETEROGENEITY_COMPLEMENTARITY_SUPPORTED:** at one early/mid timing
  (`T0`, `T0-T1`, or `T0-T2`) and one arm, both seeds have positive
  `C+F+P3 - (C+F)` AUROC and positive `C+F+P3 - (C+F+P1)` AUROC.
- **Gate C — PHENOTYPE_IS_SPATIALLY_LOCALIZED:** on the identical oracle cohort,
  at least one of CORE, PERI10, PERI20, or CORE_PERI exceeds fixed LOCAL P3 AUROC by
  `>=0.03` in both seeds for HR, HER2, subtype, or pCR at the same view/timing and
  arm.
- **Gate D — CURRENT_ENCODER_LACKS_PHENOTYPE_SIGNAL:** Gates A, B, and C all fail,
  and every seed-mean AUROC over P1/P3/P4 and CORE/PERI10/PERI20/CORE_PERI lies in
  `[0.45,0.55]` for binary HR/HER2 and macro subtype AUROC across all registered
  static views and arms. This two-sided definition does not mistake a strongly
  inverted classifier for chance.

## 9. Conditional Stage B

Stage B is authorized if and only if Gate A or Gate C passes. Otherwise it is
recorded as `NOT_RUN_NOT_AUTHORIZED`. If authorized, train exactly one seed
(`2026`) x five folds of a Response-Phenotype Dual Statistic State:

`LOCAL spatial map -> mean -> Linear(128,96)` and
`LOCAL spatial map -> population SD -> Linear(128,96)`, concatenated mean-first
and followed by one joint `LayerNorm(192)`. The output is the online
pre-projector 192-D state.

The SD forward value remains the exact weighted population SD with no epsilon.
At exactly zero variance its derivative is singular, so the preregistered
training rule uses subgradient zero (also for negative roundoff clamped to zero);
all other SD gradients use the exact square-root derivative and must be finite.

For fold `f`, use effective seed `2026+f` and construct a fresh canonical
grounded LOCAL3 model. Initialize the mean branch from rows 0--95 and the SD
branch from rows 96--191 of its `Linear(128,192)`, and copy its joint
`LayerNorm(192)`. This is functionally baseline-equivalent when SD equals mean;
all other shared modules must remain bitwise unchanged at construction. Train
the online encoder, two branches, joint norm, projector, transition, and FTV
head. Target encoder/branches/norm/projector are gradient-free EMA copies.

For parity with the canonical pilot, each fold trains on its primary outer-train
patients plus the exact 139 authorized external train-only patients; validation
and test remain the primary outer-fold partitions only. Before any Stage-B data
or model access, the content-level cache proof must authenticate all 947 C1B
caches (808 primary plus 139 external train-only), and every `train_all` ID must
belong to that proof. Authorization additionally requires a COMPLETE Stage-A
`run_summary.json` whose artifact hashes authenticate the current gates and
Stage-B authorization JSON; this run-summary SHA is carried through every
Stage-B checkpoint, selection, completion, feature, and provenance record.
The canonical C1B Stage-A data-authorization sentinel is separately bound by
its configured path and SHA-256 before loading the data contract, and that
upstream authorization path/hash is retained in Stage-B data provenance.

The sole objective is the canonical `L_JEPA + 0.25 L_FTV`, with SIGReg weight
0.09, 256 projections, and temporal weights `[2,1,0.5]`. Use AdamW, learning rate
`5e-5`, weight decay `1e-4`, physical batch 4, accumulation 8 (logical batch 32),
at most 12 epochs, patience 4, EMA 0.996, gradient clip 5, two workers, no data
augmentation, and a minimum validation representation SD of 0.05. Select the
earliest epoch minimizing validation total objective among finite non-collapsed
epochs, without test data or refitting. Export the selected online state and
repeat the exact Stage-A phenotype and causal pCR probe contract.

Every one of the 20 public Stage-B probe rows is paired one-to-one with the
Stage-A seed-2026 `LOCAL3` P1 row at the same view, target, population, and `n`.
Table 8 records the authenticated Stage-A baseline metrics and signed dual-minus-
baseline deltas for AUROC, AUPRC, and balanced accuracy; for pCR, Brier
improvement is signed as baseline minus dual so positive always means improvement.

No attention/transformer pooling, new transformer module, mask input, oracle
input, phenotype label, pCR label, or delta supervision is allowed. The existing
canonical causal transition remains the unchanged module class required by the
JEPA objective; it is not a spatial aggregator. The only auxiliary grounding
remains FTV. Frozen HR/HER2/subtype/pCR probes are then repeated. Stage B is
feasibility evidence and does not revise the Stage-A diagnosis.

## 10. Scientific classification and deliverables

The final report assigns exactly one of:

- A. `PHENOTYPE INFORMATION PRESENT BUT MEAN-POOLED AWAY`
- B. `PHENOTYPE SPATIALLY LOCALIZED`
- C. `CURRENT ENCODER LACKS PHENOTYPE INFORMATION`
- D. `MIXED`

The assignment is deterministic and uses Stage A only: choose C when Gate D
passes; otherwise choose A when Gate A or Gate B passes and Gate C does not;
otherwise choose B when Gate C passes and neither Gate A nor Gate B passes;
choose D for every remaining combination, including evidence for more than one
mechanism. Conditional Stage B cannot change this classification.

Required public tables cover the pooling contract, phenotype probes, MRI-only pCR,
clinical+FTV increments, residualized MRI, longitudinal heterogeneity, oracle
regions, and Stage B status/results. Required figures cover the pooling schematic,
phenotype and pCR AUROC, beyond-FTV delta, mean vs SD, oracle regions, longitudinal
features, and a de-identified representative activation/statistics diagnostic.
Patient-level features and predictions remain gitignored and owner-private.

The Chinese final report must explicitly answer all twelve requested questions:
whether mean pooling loses heterogeneity; whether SD has independent value;
whether mean+SD beats mean; HR/HER2/subtype effects; pCR effects; beyond-FTV
value; core-versus-peritumoral signal; oracle-versus-mask-free P3; whether the
bottleneck is pooling, localization, or encoder; whether a factorized state is
needed; whether a foundation encoder is needed; and whether Stage B was
authorized and what it found. It must link all eight tables and eight figures,
state the single Stage-A scientific classification, and record the branch,
experiment commit SHA, and GitHub push status (or `GITHUB_PUSH_FAILED`).
The delivery procedure must inspect staged files for privacy and size violations,
prove no pre-existing experiment directory changed, use the required final
experiment commit message, never force-push, and retain the local commit if a
normal push fails.
