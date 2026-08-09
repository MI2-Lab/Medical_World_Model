# Stage A5 C1B-H vs C1B-R registration sensitivity

## Result

The preregistered strategy decision is **C1B-H**: one or more preregistered registration gates failed.

The formal cohort contains 1125/1125 expected T1–T3→T0 pairs. Registration succeeded for 858 pairs and failed closed for 267 (success rate 0.7627). Failure codes: `INSUFFICIENT_OVERLAP`=2, `NONCONVERGED`=265.

## Image-only measurements

| Metric | Before median | After median | Gain/change median | Gain/change Q95 |
|---|---:|---:|---:|---:|
| Histogram MI | 0.05225680821947957 | 0.16119350896821683 | 0.09376705200510928 | 0.24089690332055214 |
| Histogram NMI | 0.013873302500517586 | 0.043100437113205964 | 0.024819765077577938 | 0.06374287355921042 |
| Whole-anatomy NCC | 0.20914047918754702 | 0.4580076229380401 | 0.21167115919272694 | 0.5021291402127639 |
| Anatomy Dice | 0.6178706206165923 | 0.7441513438994608 | 0.09913991398550476 | 0.3239153916365752 |
| Padding fraction | 0.17850073059993005 | 0.10903262870624097 | -0.049700372869318166 | 0.063863099416097 |
| Valid overlap | 0.82149926940007 | 0.890967371293759 | — | — |

Median translation magnitude was 18.645735073062585 mm and median rotation magnitude was 3.645635640752398 degrees. Catastrophic transforms numbered 17 (0.0151 of all pairs). The nonworse fraction, conservatively counting failures as not nonworse, was 0.7387.

Histogram MI is the discrete joint-histogram mutual information in nats over common automatically segmented anatomy. NMI is `MI / sqrt(H_fixed * H_moving)`. Whole-anatomy similarity is masked Pearson/NCC. All masks used for these metrics and registration were derived only from each visit's selected precontrast image.

## Preregistered gates

| Criterion | Status | Observed | Threshold |
|---|---|---:|---:|
| anatomy_lesion_residual_pattern | FAIL | 0.02564102564102564 | technical review |
| available_support_exact_containment | PASS | 0.004666666666666708 | >=-0.005 |
| blinded_technical_review | FAIL | — | technical review |
| catastrophic_transform_rate | FAIL | 0.015111111111111112 | <=0.01 |
| finite_transform_success_rate | FAIL | 0.7626666666666667 | >=0.95 |
| ftv_retention_q05 | PASS | 1.0 | >=0.95 |
| median_whole_anatomy_similarity_gain | PASS | 0.21167115919272694 | >0.02 |
| nonworse_moving_visit_fraction | FAIL | 0.7386666666666667 | >=0.75 |
| padding_increase | FAIL | median=-0.049700372869318166; Q95=0.063863099416097 | median <=0.02; Q95 <=0.05 |

The physical-support criteria use all 1,500 visits and identity/header fallback for failed registrations. The residual criterion uses all 1,125 pairs without refitting. The R-specific technical review is FAIL based on the anonymous numbered montage. No gate remains pending.

## Leakage and privacy controls

Only the `pre`, `post_early`, and `post_late` acquisition columns were accessible to phase selection, with the frozen `0/min(2,T-1)/min(5,T-1)` defaults. Registration worker payloads structurally omit FTV/localization paths, lesion measurements, clinical variables, treatment, response, and pCR. Every flagged singular visit resolves to the repaired NIfTI with no fallback to the original singular file.

Localization masks were opened only after all transforms and pair metrics were complete, solely to choose anonymous size-stratified technical-review panels. Patient IDs, paths, per-pair transforms, failure messages, and panel mappings exist only in ignored `*.private.*` files. Public CSV/JSON outputs are aggregate and the PNGs contain no identifiers.
## Post-hoc anatomy-versus-localization residual audit
No transform was refit or selected in this audit. Precontrast-derived whole-anatomy physical-centroid residuals were compared with localization-mask physical-centroid residuals after applying the already frozen transform. Localization was QC-only.

The audit completed for 1125/1125 pairs; 858 successful transforms had nonempty T0 and moving localization masks. Median anatomy residual changed from 21.004007287507697 mm to 10.445070317230915 mm. Median localization residual changed from 21.144565910307133 mm to 9.868167323216998 mm.

The literal `anatomy after >5 mm && localization after <2 mm` pattern occurred in 22 pairs (0.02564102564102564 of evaluable successful pairs); 22 additionally moved from a pre-registration localization residual at least 2 mm to below 2 mm. Status: **FAIL**. No unregistered numerical cutoff for “systematic” was invented.

Failed registrations (267 pairs) have the explicit **C1B-H identity/header fallback** disposition. A failed transform is never applied. The overall strategy remains **C1B-H**.

Public figures: `figures/03_registration_transform_distribution.png`, `figures/04_representative_t0_t3_c1b_h_vs_r.png`, and `figures/registration_anatomy_lesion_residuals.png`.
