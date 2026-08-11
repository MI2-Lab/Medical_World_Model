# MRI–Clinical Complementarity Audit — Final Report

> Diagnostic/exploratory two-seed evidence. **FORMAL AGGREGATE:** 2,000 paired bootstrap draws per cell match the configured count. The scientific boundary remains diagnostic/exploratory because only two LOCAL model seeds are available.

## Executive conclusion

The evidence-based classification is **D. CURRENT MRI STATE UNDERUTILIZES PHENOTYPE**. Profile decodability and MRI-only pCR discrimination remain weak, and no clinical-incremental or beyond-FTV timing has four entirely positive paired CIs. The best mean profile AUROC is 0.563 for HER2 at Long T0–T3 (includes late/pre-surgery); the best MRI-only pCR mean is 0.549 in `ftv_complete_375` at T3 (late/pre-surgery). This A–D assignment is a structured descriptive synthesis, not a newly tuned significance threshold.

Negative joint-model deltas are interpreted as **no supported incremental predictive value under this frozen linear protocol**, not as evidence that MRI is biologically harmful. Finite-sample estimation, high-dimensional concatenation, and regularization can make an augmented model score below its nested reference.

## Scope, estimands, and timing

| Population | n | pCR+ | Role |
|---|---|---|---|
| full_808 | 808 | 275 (34.0%) | Profile probes and secondary full-cohort C vs C+M |
| ftv_complete_375 | 375 | 110 (29.3%) | Primary fully matched C/M/F comparisons |
| ftv_unavailable_433 | 433 | 165 (38.1%) | Excluded from FTV estimand; not a comparison cohort |

The 808-patient and selected 375-patient results answer different estimands. Absolute scores across them are not paired effects. Every timing uses only its observed prefix under the [information timing contract](../information_timing_contract.csv). T3 is always labeled **late/pre-surgery** and is not silently combined with early prediction horizons. Seed 2026 and seed 3026 are model sensitivity replications on the same five patient folds, not independent patient samples.

## Patient-profile information in LOCAL MRI states

Each entry is AUROC mean (minimum–maximum) across the four seed×arm cells: 2026/3026 × LOCAL0/LOCAL3.

| Target | T0 | T1 | T2 | T3 (late/pre-surgery) | Long T0–T1 | Long T0–T2 | Long T0–T3 (includes late/pre-surgery) |
|---|---|---|---|---|---|---|---|
| HR | 0.523 (0.496–0.536) | 0.510 (0.491–0.522) | 0.546 (0.545–0.548) | 0.507 (0.474–0.540) | 0.509 (0.497–0.525) | 0.532 (0.526–0.538) | 0.534 (0.525–0.540) |
| HER2 | 0.519 (0.514–0.524) | 0.547 (0.540–0.554) | 0.549 (0.538–0.572) | 0.543 (0.518–0.557) | 0.556 (0.540–0.575) | 0.557 (0.550–0.564) | 0.563 (0.545–0.578) |
| Four-class HR/HER2 subtype | 0.515 (0.501–0.525) | 0.519 (0.512–0.526) | 0.542 (0.536–0.547) | 0.536 (0.530–0.544) | 0.532 (0.516–0.542) | 0.540 (0.534–0.552) | 0.544 (0.535–0.552) |

These are correlation/decodability probes only. Even a successful HR or HER2 probe would not establish complementarity because those biomarkers are already available to the clinical model. See [profile aggregate](../metrics/profile_oof_metrics.csv) and [profile figure](../figures/profile_auroc_heatmap.png).

## Clinical-only baselines, including treatment sensitivity

| Population | Clinical contract | Treatment | AUROC | AUPRC | Brier |
|---|---|---|---|---|---|
| full_808 | C1_hr_her2 | HR/HER2 only | 0.676 | 0.492 | 0.249 |
| full_808 | C_condition_without_treatment | without treatment | 0.661 | 0.505 | 0.212 |
| full_808 | C_condition_with_treatment | with treatment | 0.682 | 0.540 | 0.226 |
| full_808 | C2_full_without_treatment | without treatment | 0.650 | 0.487 | 0.222 |
| full_808 | C2_full_with_treatment | with treatment | 0.680 | 0.542 | 0.226 |
| ftv_complete_375 | C1_hr_her2 | HR/HER2 only | 0.660 | 0.424 | 0.249 |
| ftv_complete_375 | C_condition_without_treatment | without treatment | 0.617 | 0.431 | 0.208 |
| ftv_complete_375 | C_condition_with_treatment | with treatment | 0.681 | 0.510 | 0.225 |
| ftv_complete_375 | C2_full_without_treatment | without treatment | 0.687 | 0.453 | 0.210 |
| ftv_complete_375 | C2_full_with_treatment | with treatment | 0.697 | 0.536 | 0.243 |

Treatment arm is the assigned regimen, not delivered exposure or a causal effect. The matched with/without-treatment results are therefore prediction sensitivity analyses. Field definitions and missingness are in the [clinical feature inventory](clinical_feature_inventory.md).

## MRI-only and joint pCR discrimination

AUROC entries are mean (minimum–maximum) across four seed×arm cells.

### Full 808-patient estimand

| Timing | C | M | C+M | Clinical-error correction |
|---|---|---|---|---|
| T0 | 0.680 (0.680–0.680) | 0.520 (0.506–0.538) | 0.682 (0.669–0.702) | 0.614 (0.607–0.625) |
| T1 | 0.680 (0.680–0.680) | 0.535 (0.526–0.543) | 0.672 (0.653–0.689) | 0.629 (0.618–0.642) |
| T2 | 0.680 (0.680–0.680) | 0.523 (0.516–0.531) | 0.682 (0.662–0.703) | 0.631 (0.621–0.644) |
| T3 (late/pre-surgery) | 0.680 (0.680–0.680) | 0.546 (0.542–0.548) | 0.658 (0.638–0.679) | 0.633 (0.619–0.648) |

### FTV-complete 375-patient estimand

| Timing | C | M | C+M | F | C+F | C+F+M |
|---|---|---|---|---|---|---|
| T0 | 0.697 (0.697–0.697) | 0.512 (0.477–0.549) | 0.610 (0.575–0.632) | 0.502 (0.502–0.502) | 0.688 (0.688–0.688) | 0.600 (0.554–0.631) |
| T1 | 0.697 (0.697–0.697) | 0.521 (0.509–0.539) | 0.628 (0.625–0.631) | 0.659 (0.659–0.659) | 0.697 (0.697–0.697) | 0.650 (0.647–0.658) |
| T2 | 0.697 (0.697–0.697) | 0.530 (0.479–0.567) | 0.617 (0.602–0.658) | 0.663 (0.663–0.663) | 0.726 (0.726–0.726) | 0.645 (0.619–0.676) |
| T3 (late/pre-surgery) | 0.697 (0.697–0.697) | 0.549 (0.526–0.577) | 0.599 (0.569–0.650) | 0.691 (0.691–0.691) | 0.715 (0.715–0.715) | 0.645 (0.617–0.683) |

The other prespecified FTV benchmark, mean AUROC(C+M) − AUROC(C+F) within the same 375 patients and folds, was T0 -0.078; T1 -0.070; T2 -0.109; T3 (late/pre-surgery) -0.116. These matched-cell point differences have no dedicated bootstrap interval and are interpreted descriptively.

Full AUROC/AUPRC/Brier aggregates are in [pcr_oof_metrics.csv](../metrics/pcr_oof_metrics.csv); the primary timing plot is [here](../figures/primary_pcr_auroc_by_timing.png).

## Paired incremental effects and 95% bootstrap intervals

Positive improvement favors the augmented model for AUROC and AUPRC. Means and point ranges summarize four seed×arm cells. The displayed CI envelope is the minimum lower to maximum upper limit across four separate, fold-stratified paired patient-bootstrap 95% CIs; it is **not** a pooled CI.

| Population | Comparison | Timing | ΔAUROC mean (cell range) | AUROC 95% cell-CI envelope | Positive AUROC CIs | ΔAUPRC mean (cell range) | AUPRC 95% cell-CI envelope |
|---|---|---|---|---|---|---|---|
| full_808 | C+M − C | T0 | +0.002 (-0.011 to +0.022) | -0.049 to +0.058 | 0/4 | -0.036 (-0.052 to -0.003) | -0.101 to +0.042 |
| full_808 | C+M − C | T1 | -0.008 (-0.027 to +0.009) | -0.072 to +0.051 | 0/4 | -0.051 (-0.057 to -0.043) | -0.111 to +0.010 |
| full_808 | C+M − C | T2 | +0.002 (-0.018 to +0.023) | -0.064 to +0.065 | 0/4 | -0.023 (-0.057 to +0.009) | -0.112 to +0.067 |
| full_808 | C+M − C | T3 (late/pre-surgery) | -0.022 (-0.042 to -0.001) | -0.088 to +0.040 | 0/4 | -0.054 (-0.068 to -0.026) | -0.125 to +0.030 |
| ftv_complete_375 | C+M − C | T0 | -0.087 (-0.121 to -0.065) | -0.185 to -0.008 | 0/4 | -0.151 (-0.182 to -0.128) | -0.244 to -0.053 |
| ftv_complete_375 | C+M − C | T1 | -0.069 (-0.072 to -0.065) | -0.136 to -0.007 | 0/4 | -0.123 (-0.137 to -0.106) | -0.207 to -0.024 |
| ftv_complete_375 | C+M − C | T2 | -0.080 (-0.095 to -0.039) | -0.164 to +0.017 | 0/4 | -0.135 (-0.157 to -0.084) | -0.232 to -0.012 |
| ftv_complete_375 | C+M − C | T3 (late/pre-surgery) | -0.097 (-0.128 to -0.046) | -0.204 to +0.023 | 0/4 | -0.162 (-0.181 to -0.128) | -0.259 to -0.034 |
| ftv_complete_375 | C+F+M − C+F | T0 | -0.088 (-0.134 to -0.057) | -0.201 to +0.001 | 0/4 | -0.164 (-0.209 to -0.133) | -0.278 to -0.059 |
| ftv_complete_375 | C+F+M − C+F | T1 | -0.047 (-0.050 to -0.039) | -0.118 to +0.028 | 0/4 | -0.083 (-0.105 to -0.056) | -0.168 to +0.026 |
| ftv_complete_375 | C+F+M − C+F | T2 | -0.081 (-0.107 to -0.051) | -0.163 to +0.010 | 0/4 | -0.104 (-0.132 to -0.073) | -0.211 to +0.021 |
| ftv_complete_375 | C+F+M − C+F | T3 (late/pre-surgery) | -0.070 (-0.098 to -0.033) | -0.155 to +0.033 | 0/4 | -0.120 (-0.144 to -0.085) | -0.220 to +0.007 |

Bootstrap unit: patient within outer fold; 2,000 draws per cell in this aggregate (2,000 configured). Exact cell intervals are in [bootstrap_ci.csv](../metrics/bootstrap_ci.csv). Forest plots: [C+M vs C, full 808](../figures/full_cohort_incremental_forest.png) and [C+F+M vs C+F, selected 375](../figures/beyond_ftv_incremental_forest.png).

## MRI after removing FTV-associated components

| Timing | M | M residual | C+F | C+F+M residual | Δ residual joint − C+F |
|---|---|---|---|---|---|
| T0 | 0.512 (0.477–0.549) | 0.503 (0.481–0.529) | 0.688 (0.688–0.688) | 0.608 (0.552–0.641) | -0.080 (-0.136 to -0.047) |
| T1 | 0.521 (0.509–0.539) | 0.528 (0.513–0.548) | 0.697 (0.697–0.697) | 0.655 (0.640–0.672) | -0.043 (-0.058 to -0.025) |
| T2 | 0.530 (0.479–0.567) | 0.516 (0.500–0.534) | 0.726 (0.726–0.726) | 0.646 (0.604–0.685) | -0.080 (-0.122 to -0.041) |
| T3 (late/pre-surgery) | 0.549 (0.526–0.577) | 0.518 (0.488–0.549) | 0.715 (0.715–0.715) | 0.633 (0.587–0.678) | -0.083 (-0.128 to -0.037) |

The FTV→MRI linear map is fitted on outer train only. This residual comparison is descriptive because the paired bootstrap table is defined for raw M. See the [residual figure](../figures/residual_mri_comparison.png).

## Clinical-error test

The secondary ridge test predicts `y − p_clinical` from M using outer train, selects on validation, and corrects untouched outer-test clinical probabilities. R² tests error predictability; ΔAUROC and ΔAUPRC test whether correction improves ranking.

| Population | Timing | Error R² | Error Spearman | ΔAUROC | ΔAUPRC | Brier improvement |
|---|---|---|---|---|---|---|
| full_808 | T0 | -0.011 (-0.039–0.004) | 0.196 (0.161–0.219) | -0.066 (-0.073 to -0.055) | -0.069 (-0.093 to -0.054) | +0.009 (+0.004 to +0.012) |
| full_808 | T1 | 0.007 (-0.002–0.017) | 0.227 (0.215–0.244) | -0.050 (-0.062 to -0.038) | -0.055 (-0.062 to -0.045) | +0.013 (+0.011 to +0.015) |
| full_808 | T2 | -0.004 (-0.025–0.020) | 0.219 (0.194–0.245) | -0.048 (-0.059 to -0.035) | -0.052 (-0.066 to -0.033) | +0.011 (+0.006 to +0.016) |
| full_808 | T3 (late/pre-surgery) | 0.003 (-0.023–0.022) | 0.213 (0.179–0.244) | -0.047 (-0.061 to -0.031) | -0.045 (-0.060 to -0.031) | +0.012 (+0.007 to +0.016) |
| ftv_complete_375 | T0 | -0.027 (-0.032–-0.024) | -0.040 (-0.065–-0.021) | -0.170 (-0.180 to -0.161) | -0.214 (-0.220 to -0.206) | +0.034 (+0.033 to +0.035) |
| ftv_complete_375 | T1 | -0.031 (-0.037–-0.023) | -0.007 (-0.021–0.014) | -0.161 (-0.166 to -0.155) | -0.185 (-0.191 to -0.171) | +0.034 (+0.032 to +0.035) |
| ftv_complete_375 | T2 | -0.042 (-0.069–-0.014) | 0.038 (-0.004–0.082) | -0.122 (-0.149 to -0.097) | -0.188 (-0.212 to -0.148) | +0.031 (+0.026 to +0.037) |
| ftv_complete_375 | T3 (late/pre-surgery) | -0.053 (-0.074–-0.025) | 0.038 (0.001–0.082) | -0.123 (-0.139 to -0.101) | -0.181 (-0.203 to -0.155) | +0.029 (+0.025 to +0.035) |

A lower Brier score after correction can reflect probability shrinkage/calibration even when residual R² and discrimination do not improve; it is not by itself evidence of complementary phenotype signal. Source: [clinical_residual_metrics.csv](../metrics/clinical_residual_metrics.csv).

## Subtype-conditioned pCR findings

AUROCs are mean (minimum–maximum) across four cells. The final column is paired within cell before summarization.

| Subtype | Timing | M | Remaining clinical | Remaining clinical+M | Δ joint − clinical |
|---|---|---|---|---|---|
| HR+/HER2- | T0 | 0.556 (0.521–0.604) | 0.610 (0.610–0.610) | 0.630 (0.600–0.654) | +0.020 (-0.010 to +0.044) |
| HR+/HER2- | T1 | 0.558 (0.526–0.620) | 0.610 (0.610–0.610) | 0.617 (0.578–0.638) | +0.007 (-0.032 to +0.028) |
| HR+/HER2- | T2 | 0.511 (0.496–0.526) | 0.610 (0.610–0.610) | 0.574 (0.558–0.590) | -0.036 (-0.052 to -0.020) |
| HR+/HER2- | T3 (late/pre-surgery) | 0.497 (0.466–0.514) | 0.610 (0.610–0.610) | 0.595 (0.562–0.631) | -0.015 (-0.048 to +0.021) |
| HR-/HER2- | T0 | 0.457 (0.452–0.461) | 0.552 (0.552–0.552) | 0.469 (0.434–0.495) | -0.083 (-0.118 to -0.057) |
| HR-/HER2- | T1 | 0.463 (0.425–0.518) | 0.552 (0.552–0.552) | 0.468 (0.441–0.515) | -0.084 (-0.111 to -0.037) |
| HR-/HER2- | T2 | 0.481 (0.463–0.505) | 0.552 (0.552–0.552) | 0.501 (0.459–0.538) | -0.051 (-0.093 to -0.015) |
| HR-/HER2- | T3 (late/pre-surgery) | 0.489 (0.414–0.538) | 0.552 (0.552–0.552) | 0.502 (0.488–0.519) | -0.050 (-0.064 to -0.033) |
| HER2+ | T0 | 0.567 (0.525–0.598) | 0.649 (0.649–0.649) | 0.643 (0.618–0.661) | -0.006 (-0.031 to +0.012) |
| HER2+ | T1 | 0.540 (0.517–0.574) | 0.649 (0.649–0.649) | 0.616 (0.589–0.635) | -0.033 (-0.060 to -0.014) |
| HER2+ | T2 | 0.550 (0.533–0.584) | 0.649 (0.649–0.649) | 0.602 (0.588–0.613) | -0.047 (-0.060 to -0.036) |
| HER2+ | T3 (late/pre-surgery) | 0.554 (0.540–0.573) | 0.649 (0.649–0.649) | 0.584 (0.582–0.587) | -0.065 (-0.067 to -0.062) |

These analyses are descriptive, particularly in smaller strata, and do not create new confirmatory claims. See [subgroup aggregate](../metrics/subgroup_metrics.csv) and [subgroup figure](../figures/subgroup_auroc.png).

## Scientific classification

The evidence-based classification is **D. CURRENT MRI STATE UNDERUTILIZES PHENOTYPE**. Profile decodability and MRI-only pCR discrimination remain weak, and no clinical-incremental or beyond-FTV timing has four entirely positive paired CIs. The best mean profile AUROC is 0.563 for HER2 at Long T0–T3 (includes late/pre-surgery); the best MRI-only pCR mean is 0.549 in `ftv_complete_375` at T3 (late/pre-surgery). This A–D assignment is a structured descriptive synthesis, not a newly tuned significance threshold.

Classification logic is intentionally conservative: a profile or MRI-only result is called material only when a four-cell mean reaches AUROC 0.60 and every cell is above 0.50; an incremental timing is called supported only when all four cell-specific paired 95% AUROC CIs are above zero. These are transparent synthesis rules, not post-hoc model selection or a substitute for replication.

## Direct answers to the 12 requested questions

1. **Can LOCAL MRI predict HR?** no material signal; best mean AUROC 0.546 (0.545–0.548) at T2.
2. **Can LOCAL MRI predict HER2?** only weak signal; best mean AUROC 0.563 (0.545–0.578) at Long T0–T3 (includes late/pre-surgery).
3. **Can LOCAL MRI predict four-class subtype?** no material signal; best mean AUROC 0.544 (0.535–0.552) at Long T0–T3 (includes late/pre-surgery).
4. **How does MRI-only pCR perform?** The best mean M AUROC is 0.549 (0.526–0.577) in `ftv_complete_375` at T3 (late/pre-surgery); this is weak discrimination.
5. **How does clinical-only perform?** Primary `C2_full_with_treatment` AUROC is 0.680 in `full_808` and 0.697 in `ftv_complete_375`; the table above gives C1, original-condition, full-profile, and treatment-excluded counterparts.
6. **Does C+M beat C?** Supported timings with all four paired 95% AUROC CIs above zero: **none**. Negative cell means are read as no supported incremental value, not MRI harm.
7. **Does C+F+M beat C+F?** Supported timings: **none** in the selected 375-patient estimand.
8. **Does MRI retain pCR signal after removing FTV-associated components?** Timings where all four residual-joint AUROC deltas are positive: **none**. Clinical-error tests with uniformly positive R² and ΔAUROC: **none**.
9. **Within subtype, does MRI still distinguish pCR?** Best M mean AUROC is 0.567 (0.525–0.598) in HER2+ at T0. Uniformly positive joint-minus-clinical cells: **none**; these subgroup results remain descriptive.
10. **What is the dominant MRI contribution?** the current MRI contribution is weak/unclear and the state underutilizes phenotype.
11. **Does the current world model truly use MRI?** The original primary FLR does not establish that: its `future_response_state` is generated from lesion geometry + clinical/treatment condition, with the image transition exported separately. The LOCAL states tested here are image-only at inference, but their incremental pCR results must be supported before claiming useful MRI contribution.
12. **What image information should be strengthened next?** Prioritize a richer/foundation image encoder and pCR-free phenotype-learning objectives for morphology, heterogeneity, enhancement kinetics, and multi-scale spatial context; then repeat this frozen audit unchanged.

## Original-FLR interpretation boundary

The [clinical inventory](clinical_feature_inventory.md#critical-representation-boundary-original-state-is-not-mri-only) shows that the original frozen state used by FLR comes from geometry plus condition. The [forecast implementation](../../../ispy_jepa_tmi_clean/corejepa/models/corejepa.py#L149) and [frozen export](../../../ispy_jepa_tmi_clean/corejepa/training/runner.py#L290) do not pass the image latent into that primary readout. In contrast, the [LOCAL exporter](../../local_global_response_state_pilot/src/lg_response_pilot/features.py#L237) calls the image-only response encoder. Therefore this report distinguishes **original FLR behavior** from **LOCAL image-state behavior** and never uses the original state as M.

## Reproducibility and delivery provenance

| Item | Value |
|---|---|
| Experiment branch | feature/mri-clinical-complementarity-audit |
| Parent commit | 78ba693 |
| Evidence status | diagnostic_exploratory_two_seed_local_pilot |
| Clinical inventory SHA-256 | 50cbc417b3c60d67067f3d8c673ee9ac0e0ee200cfba9aff40a787e0ca36f262 |
| Delivery branch | feature/mri-clinical-complementarity-audit |
| Delivery commit SHA | f5386ccfb5ca9d03cc4eec4c7a60f26b588c25af |
| Push status | PUSH_SUCCEEDED |
| Remote | origin (https://github.com/MI2-Lab/Medical_World_Model.git) |
| Delivery provenance file | [delivery_provenance.json](delivery_provenance.json) |

Aggregate inputs and exact links:

- [Experiment plan](../EXPERIMENT_PLAN.md)
- [Audit configuration](../configs/audit.json)
- [Clinical feature inventory](clinical_feature_inventory.md)
- [Run summary](../metrics/run_summary.json)
- [Cohort summary](../metrics/cohort_summary.csv)
- [Clinical baselines](../metrics/clinical_baseline_metrics.csv)
- [Profile OOF metrics](../metrics/profile_oof_metrics.csv)
- [pCR OOF metrics](../metrics/pcr_oof_metrics.csv)
- [Paired bootstrap intervals](../metrics/bootstrap_ci.csv)
- [Clinical-error metrics](../metrics/clinical_residual_metrics.csv)
- [Subtype metrics](../metrics/subgroup_metrics.csv)

This generator reads no feature NPZ, OOF prediction, patient identifier, label row, or private bootstrap-draw file. All report values are computed from the linked aggregate metrics.
