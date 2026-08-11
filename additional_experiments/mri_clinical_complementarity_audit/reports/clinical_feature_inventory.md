# Clinical Feature Inventory

## Audit conclusion

The locked I-SPY2 clinical cohort contains 808 patients with complete HR, HER2, MammaPrint/MP, assigned treatment arm, and pCR; age and race each have three missing values, menopausal status has 29, and ethnicity is complete. The original CoRe-JEPA patient condition consumes only **HR, HER2, MP, age, and exact treatment arm**. Race, menopausal status, ethnicity, and the derived HR/HER2 subtype are present in the extracted table but are not loaded into `PatientRecord`, are not encoded by `ConditionEncoder`, and are not appended by the original pCR readout.

The most important representation boundary is that the original `future_response_state` is **not an MRI-only representation**. It is forecast from lesion geometry plus the clinical/treatment condition; the primary frozen landmark readout does not append an image latent. Consequently, it cannot be used as `M` in a clinical-versus-MRI complementarity comparison. This audit instead uses the LOCAL pilot's image-only online response states.

## Audited assets and cohort

| Asset | Audited value |
|---|---|
| Clinical table | `/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv` |
| Clinical table SHA-256 | `b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436` |
| Clinical table shape | 808 rows, 26 columns; 808 unique canonical `patient_id` values and 808 unique source `clinical_patient_id` values |
| pCR prevalence | 275 `label_pcr=1`; 533 `label_pcr=0` |
| Profile/probe cohort | `full_808`: all 808 locked I-SPY2 patients |
| Fully matched FTV cohort | `ftv_complete_375`: 375 patients with valid FTV at T0--T3; used for the primary matched `C/M/F` pCR comparison |
| Locked fold manifest | `/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv` |
| Fold manifest SHA-256 | `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38` |
| LOCAL MRI assets | Two model seeds (2026, 3026), five folds, `LOCAL0` and `LOCAL3`; each feature file contains 808 patients and a float32 `[808,4,192]` response-state tensor |

The configured source for the clinical table is explicit in `ispy_jepa_tmi_clean/configs/paper_v1.yaml:1-8`. The extractor reads the I-SPY2 clinical workbook and sheet at `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:17-20`, maps source records to canonical preprocessed patient keys at lines 93-107 and 132-170, and writes the four-visit subset at lines 212-226. The accompanying `/data/data/Preprocessed/I-SPY2/clinical_label_dictionary.json` describes all 985 source rows, not the 808-row complete-four-visit cohort; its counts must therefore not be copied as audit-cohort counts.

The audited source sheet has exactly ten columns: `Patient_ID`, `Arm`, `HR`, `HER2`, `MP`, `pCR`, `Age_at_Screening`, `Race`, `menopausal_status`, and `ethnicity`. The extractor carries all ten into the derived table, adding normalized/derived fields and preprocessing provenance. There is no additional source-sheet sex, stage, grade, nodal-status, BMI, separate ER/PR, delivered-treatment, or dose/adherence field to add to `C2`; none should be invented.

The relevant cohort counts are:

| Population | n | Role |
|---|---:|---|
| Raw clinical source sheet | 985 | Source population summarized by the data dictionary |
| Complete-four-visit I-SPY2 cohort | 808 | Locked clinical/profile and LOCAL feature cohort |
| FTV-complete I-SPY2 subset | 375 | Primary fully matched `C/M/F` pCR cohort |
| External I-SPY1 train-only cohort | 139 | Added only to LOCAL representation training; never evaluation |
| LOCAL technical-eligibility total | 947 | 808 I-SPY2 + 139 external train-only patients |

The LOCAL population accounting and exact-fold reuse are preregistered at `additional_experiments/local_global_response_state_pilot/configs/pilot.json:44-50`; the exported primary feature cohort is constructed at `additional_experiments/local_global_response_state_pilot/src/lg_response_pilot/features.py:237-250`.

No individual identifier is a model feature. `patient_id` is the canonical join/split key; `clinical_patient_id` and `raw_Patient_ID` are source/provenance keys only.

## Requested patient-profile fields

| Concept | Actual column(s) and source | Completeness/categories in `full_808` | Original model use | Timing and audit decision |
|---|---|---|---|---|
| HR | `label_hr`, copied from raw `HR` | Complete: 453 value 1, 355 value 0 | Loaded into `PatientRecord.hr` (`ispy_jepa_tmi_clean/corejepa/data/records.py:57-68`) and encoded into every transition condition (`ispy_jepa_tmi_clean/corejepa/data/condition.py:61-75`) | Pretreatment profile variable. Use in all clinical contracts and as a binary probe target; value 1 denotes HR+ through the subtype constructor (`ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:87-90`). |
| HER2 | `label_her2`, copied from raw `HER2` | Complete: 201 value 1, 607 value 0 | Loaded into `PatientRecord.her2` and encoded into every transition condition at the same lines as HR | Pretreatment profile variable. Use in all clinical contracts and as a binary probe target; value 1 denotes HER2+ through `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:87-90`. |
| MammaPrint / requested MP1 | `label_mp`, copied from raw `MP` | Complete: 383 value 1, 425 value 0 | Loaded into `PatientRecord.mp` and encoded as `MammaPrint` by `ConditionEncoder` (`ispy_jepa_tmi_clean/corejepa/data/condition.py:56-75`) | Pretreatment profile variable. Use in the condition-parity and full-profile baselines. The data dictionary says only “MammaPrint/MP status as provided; binary 0/1” (`ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:173-187`); report `label_mp=1` or `MP1`, and do **not** infer high-risk/low-risk or positive/negative semantics without a separate source definition. |
| Assigned treatment | `arm`, copied from raw `Arm` | Complete; 13 exact arms, listed below | Loaded into `PatientRecord.arm` (`ispy_jepa_tmi_clean/corejepa/data/records.py:57-68`), exact one-hot encoded by `ConditionEncoder` (`ispy_jepa_tmi_clean/corejepa/data/condition.py:47-75`), and also mapped to a broader routing family (`ispy_jepa_tmi_clean/corejepa/data/records.py:73-87`) | Baseline-known assigned regimen, not a response measurement. Report matched `with_treatment` and `without_treatment` analyses. It represents assigned arm only, not delivered dose, adherence, switching, or a causal treatment effect. |
| Age | `age_at_screening`, copied from raw `Age_at_Screening` | 805 observed, 3 missing; observed range 24--77 years | Loaded into `PatientRecord.age`; standardized and encoded by `ConditionEncoder` (`ispy_jepa_tmi_clean/corejepa/data/condition.py:47-75`) | Pretreatment profile variable. Imputation and scaling must be fit on outer-fold train only. Add an explicit missing indicator in the full audit baseline or document mean/median imputation. |
| Race | Prefer `race_simple`; retain `race_raw` for provenance | 805 observed, 3 missing. `race_simple`: White 647; Black or African American 88; Asian 55; Multiple 7; Native Hawaiian or Pacific Islander 5; American Indian or Alaska Native 3; missing 3 | Not a `PatientRecord` field and not used by `ConditionEncoder` or the original FLR | Available pretreatment profile variable for the extended `C2_full` baseline only. Fit categories on fold-train and handle unseen/missing values without inspecting validation/test. Normalization rules are deterministic at `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:58-66`. |
| Menopausal status | Prefer `menopausal_status_simple`; retain `menopausal_status_raw` for provenance | 779 observed, 29 missing. Premenopausal 387; Postmenopausal 258; Other_age_gt_50 64; Other_age_lt_50 41; Perimenopausal 29; missing 29 | Not a `PatientRecord` field and not used by `ConditionEncoder` or the original FLR | Available pretreatment profile variable for `C2_full`. Fit categorical handling on fold-train. The `Other_age_*` levels partly encode age, so interpret coefficients cautiously. Normalization is at `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:69-84`. |
| Ethnicity | `ethnicity`, copied from raw `ethnicity` | Complete: Not Hispanic or Latino 710; Hispanic or Latino 98 | Not a `PatientRecord` field and not used by `ConditionEncoder` or the original FLR | Available pretreatment profile variable for `C2_full`; categorical vocabulary is fold-train only. |
| HR/HER2 subtype | `hr_her2_subtype`, deterministically derived from `label_hr` and `label_her2` | Complete. HR+/HER2- 320; HR-/HER2- 287; HR+/HER2+ 133; HR-/HER2+ 68 | Not loaded by `PatientRecord`, not encoded by `ConditionEncoder`, and not directly used by the original FLR | Valid as a probe label or subgroup definition, but not an independent measurement. Do not include subtype together with HR and HER2 as predictors. The prespecified three-group analysis is HR+/HER2- (320), HR-/HER2- (287), and HER2+ (201, explicitly pooling both HER2-positive HR categories). Derivation is exact at `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:87-90,149-151`. |
| pCR target | `label_pcr`, copied from raw `pCR` | Complete: 275 value 1, 533 value 0 | Loaded into `PatientRecord.pcr` (`ispy_jepa_tmi_clean/corejepa/data/records.py:44-52`), deliberately omitted from the pretraining dataset (`ispy_jepa_tmi_clean/corejepa/data/dataset.py:88-115`), then exported/read for the frozen logistic readout (`ispy_jepa_tmi_clean/corejepa/training/runner.py:290-309`; `ispy_jepa_tmi_clean/corejepa/readout/flr.py:88-106`) | Post-treatment/post-surgery outcome. Value 1 is the positive class (`ispy_jepa_tmi_clean/corejepa/readout/flr.py:73-85`). It is the target only and is forbidden as a predictor, preprocessing input, feature-selection input, or checkpoint-selection input. |

The clinical extraction evidence for all profile fields is `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:132-170`; the field definitions are at lines 173-207. A separate preprocessing utility also carries all of HR, HER2, MP, age, arm, subtype, race, menopause, and ethnicity into a wide imaging-feature table (`ispy_jepa_tmi_clean/data_processing/preprocessing/extract_mri_nact_features.py:106-124`), but that utility does not make those fields model inputs.

## Exact treatment-arm inventory

The original condition uses the exact arm one-hot, not only the broad family. Counts in the 808-patient cohort are:

| Exact `arm` value | n | Original routing family |
|---|---:|---|
| Paclitaxel | 143 | `taxane` |
| Paclitaxel + AMG 386 | 103 | `targeted_other` |
| Paclitaxel + Neratinib | 92 | `her2_targeted` |
| Paclitaxel + Ganitumab | 82 | `targeted_other` |
| Paclitaxel + Ganetespib | 73 | `targeted_other` |
| Paclitaxel + Pembrolizumab | 61 | `io` |
| Paclitaxel + ABT 888 + Carboplatin | 59 | `platinum_parp` |
| Paclitaxel + MK-2206 | 46 | `targeted_other` |
| T-DM1 + Pertuzumab | 44 | `her2_targeted` |
| Paclitaxel + Pertuzumab + Trastuzumab | 36 | `her2_targeted` |
| Paclitaxel + Trastuzumab | 27 | `her2_targeted` |
| Paclitaxel + MK-2206 + Trastuzumab | 24 | `her2_targeted` |
| Paclitaxel + AMG 386 + Trastuzumab | 18 | `her2_targeted` |

This yields 143 `taxane`, 304 `targeted_other`, 241 `her2_targeted`, 61 `io`, and 59 `platinum_parp` primary patients. The sixth routing family, `ispy1_nact`, is for external I-SPY1 pretraining records and is absent from the 808-patient I-SPY2 evaluation cohort. The exact mapping logic is `ispy_jepa_tmi_clean/corejepa/data/records.py:73-87`; the six-family vocabulary and train-index class weighting are `ispy_jepa_tmi_clean/corejepa/data/condition.py:11-18,78-88`.

Treatment arm can encode eligibility and subtype structure, especially for HER2-targeted regimens. Improvement beyond an arm-inclusive baseline therefore answers the stricter conditional question, while the required arm-excluded counterpart shows how much signal is conditional on the assigned regimen.

## Remaining columns in the 26-column table

| Column group | Columns | Status and allowed use |
|---|---|---|
| Canonical/source identifiers | `patient_id`, `clinical_patient_id`, `raw_Patient_ID` | All complete and unique. Join, order, and provenance only; never predictors. Canonical folder mapping is built by suffix matching only when unique (`ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:93-107`). |
| Filesystem provenance | `preprocessed_dir` | Complete path string. Not read by `PatientRecord`, which constructs `root / patient_id / manifest.json` (`ispy_jepa_tmi_clean/corejepa/data/records.py:39-48`). Never a predictor. |
| Raw duplicate labels | `raw_HR`, `raw_HER2`, `raw_MP`, `raw_pCR` | Complete mirrors of normalized label columns. Never include alongside normalized fields; `raw_pCR` remains an outcome and is forbidden as a feature. |
| Raw demographic text | `race_raw`, `menopausal_status_raw` | Provenance for deterministic normalization. Prefer the normalized `*_simple` fields in analysis. |
| Imaging-availability audit | `audit_status`, `n_visits`, `complete_4visits`, `missing`, `failed_visits`, `aligned_dce_visits` | In this table all 808 rows have `audit_status=complete`, `n_visits=4`, and `complete_4visits=True`; `missing` and `failed_visits` are empty. `aligned_dce_visits` is a preprocessing diagnostic. These are selection/processing variables, not pretreatment patient profile, and are forbidden predictors. |

`PatientRecord` also attempts to read four optional columns named `mri_ld_baseline`, `mri_ld_1_3dac`, `mri_ld_interreg`, and `mri_ld_presurg` (`ispy_jepa_tmi_clean/corejepa/data/records.py:53-55`). None is present in the audited 26-column I-SPY2 table, so `longest_diameter` is unavailable there. If supplied by another cohort, follow-up LD values are time-varying imaging-response measurements, not baseline clinical fields, and must obey the same T0/T1/T2/T3 prefix restrictions as FTV.

The record's `cohort` value is injected by the loader rather than read from a clinical column (`ispy_jepa_tmi_clean/corejepa/training/runner.py:50-55`). It controls external-cohort routing; the complementarity evaluation is I-SPY2 only, so cohort must not become a predictive shortcut.

## What the original model and readout actually use

The original data path is:

1. `PatientRecord` defines only `patient_id`, `cohort`, `arm`, `hr`, `her2`, `mp`, `age`, `manifest_path`, optional `pcr`, and optional longest diameter (`ispy_jepa_tmi_clean/corejepa/data/records.py:11-24`).
2. `load_records` filters `complete_4visits`, requires an existing manifest, reads pCR and the five condition fields, and otherwise ignores the demographic columns (`ispy_jepa_tmi_clean/corejepa/data/records.py:31-70`).
3. `ConditionEncoder` builds temporal flags, exact-arm one-hot, HR, HER2, MP, and standardized age (`ispy_jepa_tmi_clean/corejepa/data/condition.py:37-75`). In the mixed I-SPY2/I-SPY1 paper cohort, this is 7 temporal + 14 arms + 3 biomarkers + age = 25 dimensions; see `ispy_jepa_tmi_clean/docs/TENSOR_CONTRACTS.md:59-71`.
4. The dataset returns image, geometry, condition, and routing target; its pretraining wrapper intentionally exposes no pCR (`ispy_jepa_tmi_clean/corejepa/data/dataset.py:73-115`).
5. The response pathway uses geometry and condition, including the non-temporal patient condition in its causal dynamics and expert gate (`ispy_jepa_tmi_clean/corejepa/models/response_state.py:33-43,97-104,124-146`).
6. The original FLR reads `future_response_state`, pCR, and split indices, constructs causal landmark-prefix features, and fits one class-balanced logistic model across three stacked landmarks (`ispy_jepa_tmi_clean/corejepa/readout/flr.py:19-63,88-129`). It does not directly append any clinical field.

Thus “not directly appended by FLR” does not mean “controlled out”: HR, HER2, MP, age, and treatment are already upstream inputs to the state passed into FLR.

The existing shortcut-audit clinical baseline confirms the intended condition-parity set. `shortcut_audit/auditlib/baseline_features.py:23-82` builds HR + HER2 + MammaPrint + fold-train-standardized age + exact-arm one-hot (plus a nominal decision code), and `shortcut_audit/auditlib/fold_evaluation.py:550-589` fits its arm vocabulary and age statistics on fold-primary-train only. Its readout uses a median imputer, scaler, and class-balanced logistic regression (`shortcut_audit/auditlib/baseline_models.py:307-326`) with train fitting and validation-only hyperparameter/threshold selection (`shortcut_audit/auditlib/baseline_models.py:392-531`). Race, menopause, ethnicity, and subtype remain absent from that parity baseline.

## Critical representation boundary: original state is not MRI-only

The clean model's primary frozen state is generated as follows:

- `CoReJEPA.forward` sends `geometry[:, :-1]` and `condition` to `response_transition` (`ispy_jepa_tmi_clean/corejepa/models/corejepa.py:126-147`).
- `forecast_response`, explicitly documented as the representation used by FLR, returns `response_transition(geometry[:, :-1], condition).future_state` (`ispy_jepa_tmi_clean/corejepa/models/corejepa.py:149-157`).
- Frozen export calls `model.forecast_response(batch["geometry"], batch["condition"])`; the image-derived transition is exported separately (`corejepa/training/runner.py:290-306`).
- The repository documentation states that the factorized response path consumes lesion descriptors plus clinical/treatment condition, and that primary FLR appends neither clinical variables nor the image latent (`ispy_jepa_tmi_clean/README.md:122-125`).

Therefore, the original `future_response_state` is a **geometry + clinical + treatment** representation. Labeling it “MRI only” would leak the very covariates the audit is supposed to condition on and would invalidate `C` versus `C+M`.

The LOCAL pilot provides the valid `M` representation:

- Clinical/treatment/pCR supervision is forbidden by `additional_experiments/local_global_response_state_pilot/configs/pilot.json:5-20`.
- Its model-facing contract contains only `image`; FTV target/mask are loss-side and geometry/mask are not model inputs (`additional_experiments/local_global_response_state_pilot/configs/pilot.json:52-59`).
- The exporter forms the exact primary train/validation/test patient order, loads only images, and calls `model.encode_response(image, None)` to write `[N,4,192]` states (`additional_experiments/local_global_response_state_pilot/src/lg_response_pilot/features.py:237-293`).
- Export metadata asserts `ftv_head_called: false` and `test_labels_used: false` (`additional_experiments/local_global_response_state_pilot/src/lg_response_pilot/features.py:294-323`).

`LOCAL0` is image-only and not FTV-grounded. `LOCAL3` is also image-only at inference, but its train-patient representation was shaped by an FTV grounding loss. It is therefore pCR-free but should not be described as unsupervised or FTV-independent.

## Target, identifier, and fold contract

The five-fold manifest has columns `patient_id,fold,split,label_pcr`, 4,040 rows, and exactly the same 808 canonical patients and pCR labels as the clinical table. Each patient occurs once in every fold and is outer test exactly once. Counts are:

| Fold | Train, pCR+ | Validation, pCR+ | Test, pCR+ |
|---:|---:|---:|---:|
| 0 | 525, 178 | 121, 42 | 162, 55 |
| 1 | 525, 179 | 121, 41 | 162, 55 |
| 2 | 525, 178 | 121, 42 | 162, 55 |
| 3 | 526, 179 | 121, 41 | 161, 55 |
| 4 | 526, 179 | 121, 41 | 161, 55 |

The manifest validation contract is implemented at `shortcut_audit/auditlib/folds.py:24-113`, with held-out and patient-order mappings at lines 128-159. The Stage-B/LOCAL loader pins the same manifest hash, reads only `patient_id`, `fold`, and `split` for representation training, validates common coverage and one test assignment per patient (`additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/contracts.py:45-47,79-88`; `additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/data.py:80-115`), and constructs train/validation/test partitions at `additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/data.py:729-768`. Thus the `label_pcr` column is available to the downstream audit but was not admitted to LOCAL model training.

The clean CoRe-JEPA implementation itself has only one seed-2026 70/15/15 pCR-stratified split (`ispy_jepa_tmi_clean/corejepa/data/records.py:90-104`; `ispy_jepa_tmi_clean/corejepa/training/runner.py:58-66`; `ispy_jepa_tmi_clean/configs/paper_v1.yaml:61-64`). It does not generate the five-fold manifest. The accessible manifest is a copied, hash-locked five-fold asset whose original generator directory is no longer available; the provenance limitation is documented at `shortcut_audit/report/repository_inspection.md:126-176`. Do not claim a particular `StratifiedKFold` algorithm beyond the observed, validated assignments.

For each seed/arm/fold, fit preprocessing and the downstream readout on that fold's `train` states, select on that fold's `val` states, and report only that fold's `test` states. Do not assemble one training matrix from each patient's test-fold representation: checkpoints from other folds can have trained on patients belonging to the current fold's held-out set. Pool only the five fold-specific held-out predictions after all models are fixed.

## Clinical baseline contract

The inventory supports these nested, prespecified contracts:

| Contract | Fields | Interpretation |
|---|---|---|
| `C1_hr_her2` | `label_hr`, `label_her2` | Minimal molecular-status baseline |
| `C_condition_without_treatment` | HR, HER2, MP, age | Exact non-temporal patient-profile variables read by the original condition encoder |
| `C_condition_with_treatment` | preceding fields + exact arm | Original condition-parity clinical/treatment baseline |
| `C2_full_without_treatment` | HR, HER2, MP, age, race, menopause, ethnicity | All usable recorded pretreatment profile fields; an audit extension beyond the original model |
| `C2_full_with_treatment` | preceding fields + exact arm | Primary full clinical/treatment baseline |

`hr_her2_subtype` is excluded from all predictor sets because it is an exact transform of HR and HER2. Identifiers, paths, raw duplicates, pCR, visit-completeness fields, alignment diagnostics, and any future MRI/FTV/LD measurement are excluded.

For every outer fold, numeric imputation/scaling, categorical vocabulary, missing indicators, feature selection, residualization, regularization selection, and thresholds must be learned from train/validation according to the frozen protocol, never from test. Categorical encoders must tolerate a validation/test category unseen in fold-train without deriving a new column from held-out data.

## Leakage and information-timing findings

1. **Original covariate preprocessing is transductive.** `ConditionEncoder(records)` is constructed from all primary train/validation/test records plus external records (`ispy_jepa_tmi_clean/corejepa/training/runner.py:69-101`), while its arm vocabulary and age mean/standard deviation are computed from every supplied record (`ispy_jepa_tmi_clean/corejepa/data/condition.py:47-59`). This is outcome-free but violates strict outer-fold isolation. The complementarity audit must refit all clinical preprocessing on each outer train split.
2. **Complete-four-visit selection uses future availability.** `load_records` filters on `complete_4visits` (`ispy_jepa_tmi_clean/corejepa/data/records.py:31-43`), and the extractor creates this subset at `ispy_jepa_tmi_clean/data_processing/preprocessing/extract_clinical_labels.py:223-224`. T0 results therefore describe patients who later completed all four imaging visits, not an unselected baseline-deployment population. This is a cohort-selection limitation even when the T0 feature vector itself is causal.
3. **Treatment is a condition, not an outcome or causal exposure estimate.** Exact assigned arm is available at prediction time under the audit contract, but it must be labeled `with_treatment`; the arm-excluded analysis is required. No claim about treatment efficacy or delivered exposure follows from its coefficient.
4. **Observed MRI must use a prefix.** LOCAL states are `[T0,T1,T2,T3]`. T0 uses only index 0; T1 uses indices 0--1; T2 uses 0--2; and T3 uses 0--3. T3 is a late pre-surgery assessment and is reported separately. Never compute an early feature from a mean, delta, or concatenation that includes a later visit.
5. **FTV/LD must use the same prefix.** The raw FTV adapter reconstructs four static values from adjacent transitions (`additional_experiments/g3_multiseed_generalization/src/dgrs/data.py:56-100`). At timing `Tk`, only FTV through `Tk` is admissible. Future FTV or follow-up LD in an earlier model is direct future-information leakage.
6. **pCR enters only downstream.** The clean pretraining wrapper excludes pCR and the LOCAL checkpoint contract forbids it. Reading pCR for split validation, downstream train/validation fitting, and final held-out scoring is allowed; using it for MRI feature construction, upstream checkpoint choice, preprocessing, or residualization is not.
7. **Bootstrap patients, not visits.** Each patient contributes repeated timepoints and appears in five manifest rows. Paired uncertainty must resample canonical patients while preserving the compared predictions and fold stratum; visits and model seeds are not independent samples.
8. **Two seeds are sensitivity replications.** Seeds 2026 and 3026 share the same patient folds. Report seed-specific and paired summaries; do not multiply the clinical sample size by two.
9. **FTV-complete restriction changes the estimand.** The 375-patient matched `C/M/F` analysis is selected on complete FTV availability. Report the 808-patient profile and secondary `C` versus `C+M` analysis separately, and never present an absolute difference between 808- and 375-patient results as an incremental model effect.

## Final interpretation boundary

An HR/HER2/subtype probe establishes only that LOCAL states contain image information correlated with molecular phenotype. The conditional scientific endpoints remain paired held-out changes for `C+M` versus `C` and `C+F+M` versus `C+F`, with identical patients, timing, folds, target, and preprocessing protocol. The original geometry-and-condition `future_response_state` cannot supply that evidence; only the fold-matched image-only LOCAL states can serve as `M`.
