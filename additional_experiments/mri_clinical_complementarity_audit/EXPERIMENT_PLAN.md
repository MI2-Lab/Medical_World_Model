# MRI–Clinical Complementarity Audit

## Scientific question

This audit asks whether the frozen longitudinal DCE-MRI response state contains information beyond established patient-profile variables—especially HR/HER2, treatment, and FTV/tumor burden—and quantifies its incremental value for pCR prediction. It does **not** redesign or retrain the encoder, LOCAL pooling, JEPA transition, or FTV grounding objective.

The evidence is diagnostic/exploratory. It uses the two completed LOCAL pilot training seeds (2026 and 3026), five locked outer folds, and both `LOCAL0` and `LOCAL3`. Five folds are paired held-out partitions, not independent training replications. Any later multi-seed LOCAL confirmation must replicate these conclusions separately.

## Frozen provenance

- Branch: `feature/mri-clinical-complementarity-audit`
- Parent commit: `78ba693` (`Add local-global response state pilot`)
- MRI tensors: selected-checkpoint online pre-projector response states, shape `[N,4,192]`
- Arms: `LOCAL0`, `LOCAL3`
- Seeds: `2026`, `3026`
- Folds: the exact seed-2026 patient manifest used by the LOCAL pilot (SHA-256 locked in `configs/audit.json`)
- No pCR, HR, HER2, subtype, race, age, treatment, or test labels were used to select the upstream LOCAL checkpoints.

## Cohorts and matched estimands

Two prespecified cohorts are kept separate:

1. `full_808`: all 808 I-SPY2 patients in the locked folds. This is used for HR/HER2/subtype probes and the secondary `C` versus `C+M` pCR analysis.
2. `ftv_complete_375`: the 375-patient intersection with valid FTV at T0–T3. This is the **primary fully matched pCR cohort**. Every model in the headline table—`C`, `M`, `F`, `C+F`, `C+M`, `C+F+M`, `M_residual`, and `C+F+M_residual`—uses these same patients, folds, labels, and held-out protocol.

No result may compare absolute scores across these populations as if they were a paired incremental effect.

## Information timing

The four independent prediction timings are T0 (pre-NAC), T1 (early NAC), T2 (inter-regimen), and T3 (pre-surgery). T3 is reported separately as a late response assessment and is never silently merged with earlier prediction horizons.

- MRI at timing `Tk`: concatenate only observed states `r_T0 ... r_Tk`.
- FTV at timing `Tk`: use only `log1p(FTV_T0) ... log1p(FTV_Tk)`.
- Clinical profile: pretreatment variables only.
- Treatment-arm analyses are explicitly labeled `with_treatment` or `without_treatment`.
- The machine-readable contract is `information_timing_contract.csv`.

## Clinical contracts

- `C1_hr_her2`: HR and HER2.
- `C_condition_without_treatment`: HR, HER2, MammaPrint/MP1, and age—the exact non-temporal patient-profile fields consumed by the original condition encoder.
- `C_condition_with_treatment`: the preceding fields plus exact treatment arm.
- `C2_full_without_treatment`: every usable recorded pretreatment profile field (HR, HER2, MP1, age, race, menopausal status, and ethnicity).
- `C2_full_with_treatment`: the preceding fields plus exact treatment arm. This is `C` in the primary incremental analysis.

Categorical vocabularies, missing-value handling, numeric imputation, and scaling are fitted on fold-train only. Unknown validation/test categories map to the all-zero one-hot block; test data never determines the vocabulary.

## Fold-safe linear evaluation

All primary readouts are L2 logistic regressions. For each outer fold and endpoint:

1. preprocessing is fitted on fold-train only;
2. each `C` candidate is fitted on fold-train only;
3. `C` is selected by validation AUROC, with smaller `C` as the deterministic tie-break;
4. the selected train-fitted model is evaluated once on outer test; and
5. no model is refit on test or tuned using test metrics.

Binary profile probes select their decision threshold by validation balanced accuracy. Multiclass subtype uses the model's argmax and macro one-vs-rest AUROC/AUPRC. pCR endpoints report AUROC, AUPRC, and Brier score; thresholds are not optimized on test.

## Representation probes

`LOCAL0` and `LOCAL3` are separately probed for:

- HR and HER2 at T0, T1, T2, and T3;
- the observed four-class dataset subtype (`HR+/HER2-`, `HR-/HER2-`, `HR+/HER2+`, `HR-/HER2+`); and
- longitudinal prefix representations T0–T1, T0–T2, and T0–T3.

A successful probe means the state contains molecular-phenotype-correlated information. It does not by itself establish complementarity because the same variables are already available clinically.

## pCR model families

At each timing the primary 375-patient table evaluates:

- `C`: full pretreatment clinical profile plus treatment;
- `M`: frozen LOCAL prefix;
- `F`: observed FTV prefix;
- `C+F`, `C+M`, and `C+F+M`;
- `M_residual`: MRI after removing the fold-train linear component associated with the available FTV prefix; and
- `C+F+M_residual`.

The primary effects are:

`delta_MRI_incremental = AUROC(C+M) - AUROC(C)`

`delta_MRI_beyond_FTV = AUROC(C+F+M) - AUROC(C+F)`

The same paired changes are also reported for AUPRC and Brier score (for Brier, lower is better and the effect is oriented as baseline minus augmented).

## Residual and subgroup analyses

- FTV residualization fits `FTV prefix -> each MRI dimension` on outer train only and applies that frozen mapping to validation/test.
- A clinical-error probe fits a fold-train clinical model, models `y - p_clinical` from MRI on fold-train, selects ridge strength on validation, and evaluates the untouched outer-test residual. The direct `C+M` versus `C` comparison remains the primary conditional test.
- Subtype analyses use dataset labels to form three prespecified clinical strata: `HR+/HER2-`, `HR-/HER2-`, and `HER2+` (the latter combines the two observed HER2-positive HR categories). Within each stratum, MRI-only, remaining-clinical-only, and their joint model are fit with the same outer-fold isolation. Small subgroup results are descriptive.

## Uncertainty and classification

Paired confidence intervals resample patients—not visits—within held-out fold strata. Bootstrap draws preserve each model pair and compute changes from the same sampled patients. The scientific conclusion will be assigned to one of the requested A–D categories, based on consistency across arms, seeds, timings, and paired intervals rather than one favorable point estimate.
