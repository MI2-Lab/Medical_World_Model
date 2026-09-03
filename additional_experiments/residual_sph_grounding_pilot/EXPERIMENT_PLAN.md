# Goal S1: FTV + residual-SPH grounding pilot

## Status and scientific question

This is a new, exploratory, two-seed representation-learning pilot. It asks whether fold-safe supervision with the FTV-independent component of sphericity (SPH) can organize the confirmed LOCAL response state toward non-volume morphology while preserving its established static FTV and observed delta-FTV behavior. Representation learning and checkpoint selection are pCR-free; pCR is opened only after a cryptographic representation freeze.

SPH is called a **non-volume morphological measurement**. A successful residual target may be called an **FTV-independent morphological component under the registered linear residualization convention**. Neither term means molecular phenotype, biological subtype, treatment mechanism, causal independence, or clinical utility.

The scientific protocol in this plan and `configs/pilot.json` is frozen before formal training results. The single sensitivity at `lambda_sph=0.10` is descriptive and cannot replace or retune the `0.05` primary result.

## Evidence boundary

The reviewed evidence is summarized with exact hashes in `manifests/prior_evidence_summary.md`.

- The five-seed study concluded `LOCAL_MULTISEED_CONFIRMED`. LOCAL3 is the current grounded LOCAL response-state reference.
- LOCAL3 minus LOCAL0 static FTV macro Spearman effects were +0.025563, +0.041267, +0.022294, +0.019178, and +0.040023 across seeds 2026-6026 (mean +0.029665; bootstrap 95% CI +0.021701 to +0.037629). Observed delta-FTV effects were +0.033576, +0.038419, +0.040932, -0.010990, and +0.059287 (mean +0.032245; 95% CI +0.008774 to +0.049971). Optimization safety was 25/25. This supports a small, stable FTV-grounding effect, not a large phenotype claim.
- Classical tabular evidence showed joint non-FTV downstream relevance, with LD descriptively strongest, but did not prove a separate stable LD-residual contribution. The subsequent image audit failed raw-LD and residual-LD gates. Downstream/tabular relevance therefore did not translate into stable FTV-independent LD decodability from the current image state.
- BPE remains blocked: its contralateral central-five-slice FGT source does not have verified coverage inside the lesion-centered 64-mm LOCAL field. Source/FOV provenance, occupancy, and boundary-touch evidence are missing (`FOV_OBSERVABILITY_UNVERIFIED`).
- The earlier SPH residual gate passed at T0 only for `LOCAL3/Z4`, the full-local mean+standard-deviation spatial statistic (256 dimensions): residual Spearman 0.314970/0.311879 and reconstructed-target R2 0.369936/0.370913 in seeds 2026/3026. It did **not** stably pass in the current 192-dimensional response state `Z2`: Spearman 0.236397/0.098652 and residual-space R2 0.026614/-0.076541. This motivates an experiment; it does not establish that direct grounding will work.
- Dynamic residual-SPH evidence was weak and failed. No delta-SPH supervision is allowed.
- The previous MRI-clinical audit found weak MRI-only pCR discrimination and no stable beyond-FTV complementarity; its mean `C+F+M - C+F` AUROC effects in the 375-person estimand were -0.088, -0.047, -0.081, and -0.070 from T0 through T3. These are post-freeze comparison anchors only.

The evidence directories are workspace-local and are not tracked by this branch's starting commit (`7644e3835af6b12899c57819bedd1876572c434f`). S0 confirmation outputs therefore have an ancestry caveat: the formal resolver must verify the expected experiment, seed, fold, selected-epoch metadata, paired initialization, source hashes, and per-file SHA-256 values. An untracked file with a plausible name is never sufficient. Missing or mismatched S0 provenance stops the affected cell.

## Frozen base contract

The following remain unchanged from confirmed LOCAL3:

- C1B-H, DCE7, the technical-eligibility contract, the exact 375-person complete-case SPH estimand, and the existing five outer folds;
- the 3-D encoder, source feature block, fixed 64 x 64 x 64 mm physical LOCAL support, fractional-overlap LOCAL pooling, and 192-dimensional current response state;
- image geometry `float32 [B,4,7,112,176,160]`, spatial shape `[112,176,160]` in ZYX order, and spacing `[0.9,0.9,2.0]` mm in XYZ order;
- JEPA, EMA target encoder, SIGReg, AdamW, the training schedule, FTV target, and `lambda_ftv=0.25`;
- fixed seed/fold pairing, patient order, shared-module initialization, and stochastic stream;
- no lesion-mask readout, no adaptive pooling, no geometry redesign, no encoder replacement, no global branch, and no high-capacity SPH-specific network.

The confirmed transition is image-only. The source request's “treatment/clinical transition-conditioning contract” is interpreted as an instruction to preserve the confirmed contract, not as permission to introduce treatment or clinical fields. Image tensors are the only model inputs. FTV and SPH are loss-side targets. Treatment, HR, HER2, other clinical variables, and pCR are inaccessible to representation training.

The JEPA population construction also remains unchanged. Each fold uses the broader frozen Stage-B training construction, including its fixed external-train-only candidates, rather than shrinking all representation learning to 375 patients. SPH loss is endpoint-masked to valid same-visit SPH in the exact 375-person complete-case population. A patient or visit without SPH may still contribute to frozen JEPA and separately valid FTV losses. This preserves the base estimand and avoids treating missing auxiliary labels as zeros.

## Exact SPH target and fold-safe residualization

The only SPH fields are `SPHERICITY_T0`, `SPHERICITY_T1`, `SPHERICITY_T2`, and `SPHERICITY_T3`. They are the dimensionless ratio of the surface area of an equal-volume sphere to the surface area of the three-dimensional FTV tumor mask. No alternative segmentation, radiomics implementation, field, or recalculation is allowed.

Residualization is performed independently for each outer fold and visit, using the primary outer-training patients only:

1. Fit 1st/99th percentile winsor limits separately for FTV and SPH on the outer-training observations at that visit.
2. Clip with those train-fitted limits. Apply `log1p` to FTV and identity to SPH.
3. Standardize transformed FTV and SPH separately with their outer-training mean and population standard deviation.
4. Fit `Ridge(alpha=1, fit_intercept=True)` from standardized FTV to standardized SPH.
5. Define `epsilon = SPH_std - Ridge_train(FTV_std)`.
6. Fit a second outer-training mean and population standard deviation to `epsilon`; the S2 grounding target is this standardized residual.

This is the prior Goal-6 convention. The shorthand `SPH_res = SPH - f_train(FTV)` does **not** authorize subtraction in raw natural units. S1 uses the train-standardized, winsorized identity SPH from step 3. S2 uses the twice-standardized residual from step 6. Validation and test use only the corresponding train-fitted limits, transforms, scales, ridge parameters, and residual scale. The residualizer never sees validation/test membership during fit and never reads pCR, HR, HER2, clinical fields, or treatment.

Each fold/visit coefficient record must contain sample count, winsor limits, transform names, means/scales, ridge intercept/coefficient, residual mean/scale, input hashes, and fit-scope assertions, but no patient identifier or row-level value. Reconstruction combines a probe-predicted residual with the conditional SPH component computed from observed same-visit FTV under that fold's frozen transform and then inverts the SPH scaling.

## Arms and matrix

Seeds are 2026 and 3026; folds are 0-4. The training effective seed is `seed_base + fold`. The formal matrix is paired by seed and fold.

| Arm | Purpose | Objective | Formal cells |
|---|---|---|---:|
| S0 | Hash-verified confirmed LOCAL3 reference | `L_JEPA + 0.25 L_FTV + existing regularization` | 10 reused reference cells |
| S1 | Raw-SPH mechanistic control | `L_JEPA + 0.25 L_FTV + 0.05 L_SPH + existing regularization` | 10 new cells |
| S2 | Primary residual-SPH candidate | `L_JEPA + 0.25 L_FTV + 0.05 L_SPH_res + existing regularization` | 10 new cells |
| S2-L10 | One residual-SPH sensitivity | same, with `lambda_sph=0.10` | 10 new descriptive cells |

The S1/S2 head is exactly one shared `Linear(192,1)` layer applied to each concurrent visit's current LOCAL response state. There is no timing-specific MLP. Loss is Huber with `delta=1.0` after the registered train-fold target standardization.

All four same-visit targets are supervised when valid: `state_T0 -> SPH_T0`, ..., `state_T3 -> SPH_T3`. This four-visit use is a **new extrapolation**, because the prior stable signal was T0 `LOCAL3/Z4`, not every visit and not the current response vector. Gate B consequently remains T0-only. No difference, percent change, next-visit target, future SPH, or longitudinal SPH target enters optimization.

## Optimization and checkpoint selection

Frozen optimization settings are physical batch 4, eight-way accumulation to logical B32, one exact nonlinear SIGReg reduction per logical batch, one gradient clip/AdamW step/EMA update per logical batch, AdamW learning rate `5e-5`, weight decay `1e-4`, SIGReg weight `0.09`, JEPA step weights `[2,1,0.5]`, EMA momentum `0.996`, maximum gradient norm 5, 12 epochs, and patience 4. The collapse floor is validation representation standard deviation 0.05.

Selection deliberately does not reward validation SPH. For each S1/S2/S2-L10 seed-fold cell:

1. Resolve the selected, hash-verified paired S0 checkpoint and its validation state loss.
2. An epoch is eligible only if all monitored values are finite, the representation is noncollapsed, and validation state loss is at most `1.05 * paired_S0_selected_validation_state_loss`.
3. Among eligible epochs, choose minimum validation FTV loss; ties use lower validation state loss and then earlier epoch.
4. If no epoch is eligible, choose the smallest positive state-gate violation, then validation FTV loss, validation state loss, and earlier epoch, and mark the cell failed.

Neither validation SPH loss nor any test endpoint may select a checkpoint. pCR, clinical variables, treatment, test FTV/SPH, observed delta FTV, and every dynamic SPH measurement are forbidden selection inputs. Early stopping follows the same pCR-free, SPH-blind selection ordering. This strict base-compatibility selector tests whether SPH can reorganize state without allowing the auxiliary target to pick a favorable epoch.

S0 must reproduce the confirmed behavior through verified artifacts. If reconstruction rather than reuse becomes technically necessary, it requires a separately documented provenance amendment made before viewing S1/S2 formal outcomes; it cannot be silently substituted under this lock.

## Frozen representation evaluation

After checkpoint selection, the networks are frozen. Each probe fits on the outer-training split, selects ridge alpha from `[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]` by minimum validation target-space MSE with smaller-alpha tie break, does not refit on train+validation, and predicts the outer test once. Five held-out folds are pooled into one OOF result per training seed; seeds, not folds, are the independent replication units.

### Response preservation

For static FTV at T0-T3 and the unweighted timing macro, report Spearman, Pearson, natural-unit R2, RMSE, MAE, and prediction/target variance ratio. For observed delta FTV, use literal natural-unit adjacent changes with `state_end - state_start`, the three adjacent intervals, and their unweighted macro; report the same metrics. Delta FTV remains evaluation-only.

### SPH organization and burden redundancy

Probe frozen state to both raw SPH and standardized `SPH_res` at each visit. Report Spearman, Pearson, residual-space R2 where valid, reconstructed-target R2, RMSE, MAE, and seed consistency. The primary phenotype endpoint is T0 `SPH_res` Spearman.

Also report FTV-to-state, SPH-to-state, and SPH-res-to-state decodability and the partial correlation between state-derived SPH and target SPH controlling same-visit FTV. If raw SPH improves without SPH-res improvement, the interpretation is burden-redundant improvement, not beyond-FTV morphology.

Longitudinal SPH behavior may be described only after freeze and must be labeled post hoc. It cannot enter Gates A-C, modify the checkpoint, or be reframed as supervised delta-SPH evidence.

## pCR firewall and post-freeze evaluation

The representation executable must be structurally unable to import or resolve the clinical-label input. Before any pCR process starts, a representation-freeze manifest must hash all selected checkpoints, residualizers, transforms, exported states, representation-probe specifications, and completed pCR-firewall audit. Changing any representation artifact invalidates the post-freeze analysis.

Only the separate post-freeze process reads pCR and the clinical contract. It evaluates `M`, `C`, `C+M`, `C+F`, and `C+F+M` under the prior fold-safe protocol. `C` is the frozen full baseline/treatment contract: HR, HER2, MP, age at screening, race, menopausal status, ethnicity, and assigned treatment arm. Their use is evaluation-only and does not contradict the image-only representation contract.

At each timing prefix (T0, T0-T1, T0-T2, and descriptive T0-T3), preprocessing is fit on outer train, L2 `liblinear` logistic regression selects `C` from `[1e-4,1e-3,1e-2,1e-1,1,10,100]` by validation AUROC with smaller-C tie break, train+validation refit is forbidden, and outer test is predicted once. Paired AUROC comparisons use 2,000 patient resamples stratified within outer fold, seed 260811, with 95% percentile intervals. pCR cannot choose S1 versus S2, an epoch, a target transform, or lambda.

## Registered comparisons and gates

The paired comparisons are:

- E1: S2-S0 static FTV macro Spearman.
- E2: S2-S0 observed delta-FTV macro Spearman.
- E3: S2-S0 T0 SPH-res Spearman.
- E4: S2-S1 T0 SPH-res Spearman.
- E5: S2-S0 MRI-only pCR AUROC at each frozen timing.
- E6: `C+F+M_S2 - C+F`, compared with `C+F+M_S0 - C+F`.

Thresholds below are descriptive decision thresholds, not p-values.

### Gate A: response safety

Pass `RESPONSE_STATE_PRESERVED` only if all conditions hold:

- E1 is at least -0.03 in both seeds;
- observed delta FTV does not show systematic degradation, operationalized before training as **not both seed-level E2 effects being strictly negative**; and
- at least 9/10 primary S2 (`lambda=0.05`) seed-fold cells satisfy the registered state-loss eligibility rule without fallback.

All observed delta magnitudes are reported even when the direction rule passes. S1 and S2-L10 safety are reported but do not replace the primary 9/10 calculation.

### Gate B: residual-SPH organization

At T0, E3 must be strictly positive in both seeds and its two-seed mean must be at least +0.05. This minimum pass is `RESIDUAL_SPH_GROUNDING_WORKS`. The strong form additionally requires mean E3 at least +0.10. T1-T3 cannot rescue a T0 failure.

### Gate C: benefit over raw SPH

E4 must be strictly positive in both seeds. Passing gives `RESIDUAL_TARGET_IS_PREFERABLE`. Failure means any gain cannot be attributed specifically to residualization.

### Gate D: downstream complementarity

At one same eligible timing among T0, T0-T1, or T0-T2, `C+F+M_S2 - C+F` must be strictly positive in both seeds. Passing gives `SPH_GROUNDED_STATE_ADDS_BEYOND_FTV`. The strong form additionally requires the two-seed mean AUROC increment at a qualifying timing to be at least +0.03. Gate D is post-freeze only and never retroactively affects training or selection.

## Classification hierarchy

To make the source classifications mutually exclusive, apply this order:

1. If A or B fails: `SPH_GROUNDING_NOT_USEFUL`; return to LOCAL3.
2. Otherwise, if C fails: `RAW_SPH_SUFFICIENT`; do not claim beyond-FTV morphology, regardless of a downstream descriptive result.
3. Otherwise A+B+C pass: `RESIDUAL_SPH_GROUNDING_VALIDATED`, which is the condition that justifies five-seed confirmation.
4. Within validated results, if D also passes, add `RESIDUAL_SPH_COMPLEMENTARITY_SUPPORTED`; if D fails, classify the downstream interpretation as `MORPHOLOGY_ORGANIZED_BUT_NO_PCR_COMPLEMENTARITY` and retain SPH, at most, as a morphology auxiliary.

The `lambda=0.10` sensitivity cannot change this hierarchy or authorize confirmation.

## Required outputs

Public outputs contain no row-level data:

- target-transform and residualizer audits, including one coefficient record per fold/visit;
- optimization trajectories and the 10-unit S2 safety table;
- static FTV and observed delta-FTV tables;
- raw SPH/SPH-res, reconstruction, redundancy, and partial-correlation tables;
- pCR model and paired-bootstrap tables produced only after freeze;
- seed-level effects, aggregate figures, `metrics/decision.json`, and a Chinese `reports/final_report.md` answering all 12 requested questions;
- a run manifest recording branch, final commit SHA, push command/status, data/code hashes, and `GITHUB_PUSH_FAILED` if delivery fails.

Patient identifiers, row-level predictions, feature arrays, checkpoints, raw MRI, the source workbook, and private absolute paths are forbidden from git. Private runtime artifacts use ignored directories and `.private.` names with mode 0600 where supported. Before commit, the staged tree must pass the checks in `manifests/privacy_contract.json`.

