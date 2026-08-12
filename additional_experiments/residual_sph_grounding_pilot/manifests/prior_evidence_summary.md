# Prior evidence and scope boundary

This file records the evidence that motivated the residual-SPH pilot. It is not a claim that the new intervention already works. The four evidence directories were available in the working tree and were reviewed before the new protocol was frozen. They are not tracked by the starting commit of this branch; therefore local presence alone is not provenance. The cited reports and machine-readable tables are hash-anchored below, and any S0 artifact used by the formal run must additionally pass the cell-level ancestry checks described in the plan.

## 1. LOCAL response-state confirmation

The five-seed confirmation classified the architecture as `LOCAL_MULTISEED_CONFIRMED`. LOCAL3 is therefore the current grounded LOCAL response-state reference for this pilot. This wording does not invent a separate historical label such as `LOCAL3_REFERENCE_LOCKED`; it describes how the confirmed arm is used here.

FTV grounding produced a small but stable gain over LOCAL0:

| LOCAL3 - LOCAL0 effect | 2026 | 3026 | 4026 | 5026 | 6026 | mean | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static FTV macro Spearman | +0.025563 | +0.041267 | +0.022294 | +0.019178 | +0.040023 | +0.029665 | +0.021701 to +0.037629 |
| Observed delta-FTV macro Spearman | +0.033576 | +0.038419 | +0.040932 | -0.010990 | +0.059287 | +0.032245 | +0.008774 to +0.049971 |

All five static effects and four of five observed-delta effects were positive. Optimization safety passed in 25/25 matched folds. For the two seeds reused here, the confirmed LOCAL3 five-fold pooled OOF anchors were static FTV macro Spearman 0.530882 (2026) and 0.513249 (3026), and observed delta-FTV macro Spearman 0.340132 and 0.300194.

The confirmed implementation is image-only: the transition receives image-state inputs and no treatment or clinical covariates. The source goal's instruction to preserve the “treatment/clinical transition-conditioning contract” is therefore resolved by preserving that confirmed absence; this pilot must not add treatment or clinical inputs.

## 2. Classical DCE phenotype complementarity

The classical audit supported joint non-FTV information (`N = LD + SPH + BPE`, and its FTV-residualized form) downstream of clinical+FTV in its own tabular setting. Its family ablation localized the largest descriptive contribution to LD, but did not establish a separate stable `LD_res` pCR increment. This distinction matters: tabular/downstream relevance is not proof that the current MRI response state can decode an FTV-independent target.

## 3. Non-FTV image-decodability audit

The image audit did not support LD as a robust new grounding target. Raw LD failed its registered observability gate and FTV-residual LD failed its beyond-FTV gate. LD can remain a burden/extent descriptor, but the combination of classical relevance and these image results does not show stable FTV-independent LD decodability from the current state.

BPE is not eligible for this experiment. The source is contralateral-breast, central-five-slice fibroglandular-tissue early enhancement, whereas the current LOCAL state has fixed lesion-centered 64-mm support. Source-side/FOV mapping, crop occupancy, and boundary-touch provenance were unavailable, so its status remains `FOV_OBSERVABILITY_UNVERIFIED`; numerical probe results cannot authorize BPE grounding.

SPH requires an equally important qualification. The prior registered SPH residual gate passed only at T0 for `LOCAL3/Z4`, the full-local spatial mean+standard-deviation representation with 256 dimensions:

| T0 FTV-residual SPH result | Seed 2026 | Seed 3026 |
|---|---:|---:|
| `LOCAL3/Z4` residual Spearman | 0.314970 | 0.311879 |
| `LOCAL3/Z4` residual-space R2 | 0.072767 | 0.083717 |
| `LOCAL3/Z4` reconstructed-target R2 | 0.369936 | 0.370913 |

That result was not a stable pass for the current 192-dimensional response state `Z2`:

| T0 FTV-residual SPH result | Seed 2026 | Seed 3026 |
|---|---:|---:|
| `LOCAL3/Z2` residual Spearman | 0.236397 | 0.098652 |
| `LOCAL3/Z2` residual-space R2 | 0.026614 | -0.076541 |
| `LOCAL3/Z2` reconstructed-target R2 | 0.337088 | 0.270718 |

Thus the prior audit supplies evidence that local spatial statistics contain T0 residual-SPH information, not evidence that the confirmed response vector already represents it reliably. Direct residual-SPH grounding is a new, explicitly exploratory intervention intended to test whether that information can organize the response state. It was not automatically authorized or recommended by the prior audit, whose overall SPH classification remained `MIXED OR UNRESOLVED`.

Dynamic evidence failed. The SPH-specific dynamic scorecard selected `LOCAL0/Z4` but had no qualifying early interval (minimum-seed early macro residual Spearman about 0.008 and reconstructed R2 about 0.018). No delta-SPH, percent-change-SPH, or future-SPH target is permitted here.

## 4. MRI-clinical complementarity audit

The earlier two-seed LOCAL states did not show stable pCR complementarity. In the 375-person estimand, mean `C+F+M - C+F` AUROC effects were -0.088, -0.047, -0.081, and -0.070 at T0, T1, T2, and T3 respectively. MRI-only discrimination was weak, and no timing supported a beyond-FTV increment. These are prior negative anchors, not training objectives. The new representation must be frozen before the pCR table is read.

## Consequence for this pilot

The experiment is restricted to same-visit static SPH supervision. T0 is the primary SPH gate because that is where the prior `LOCAL3/Z4` result passed. Training a shared head at T1-T3 is a new extrapolation, retained only to test concurrent morphology across the four already frozen visits; those visits cannot substitute for T0 in Gate B. The primary test is whether fold-safe, standardized FTV-residual SPH supervision can transfer the spatial-statistic signal into the current LOCAL response state without eroding the confirmed FTV/observed-delta-FTV representation.

## Evidence hashes

| Evidence artifact | SHA-256 | Tracked in starting commit? |
|---|---|---|
| `local_response_state_multiseed_confirmation/reports/final_report.md` | `6f495ea2412b3fe6d6e24fa9c9f067f89685fa7e69c40758582f11d2855835da` | no |
| `local_response_state_multiseed_confirmation/metrics/decision_summary.json` | `40906f527381fd67e2fb8c9190a894fde8a63a6cf1c72ffc4b4c8f64fc693a1a` | no |
| `local_response_state_multiseed_confirmation/metrics/table2_static_ftv.csv` | `cff3301e931759ea993c269ea613c22cc1df74a05a1f8d58f3565ce6f21469e0` | no |
| `local_response_state_multiseed_confirmation/metrics/table3_observed_delta_ftv.csv` | `a49b3675cd1e614ba367685ba3476a945070f45ce8b7d387fd021cacc7a89f68` | no |
| `classical_dce_phenotype_complementarity/reports/final_report.md` | `c9b1d5a6b4e7ea1cb9b8eb5002fcfe281b5fec36ae79fea7cf259356b69dfd0b` | no |
| `nonftv_phenotype_decodability_audit/reports/final_report.md` | `96a879f0655beca4a280a499e3ca78404f338b93b5e9fd7e9298d698052de915` | no |
| `nonftv_phenotype_decodability_audit/metrics/primary_gates.json` | `11907a98261048a0629ce2ff041ad3b6e95c5d8b3b489808d3a1c3d663084811` | no |
| `nonftv_phenotype_decodability_audit/metrics/gate_candidate_matrix.csv` | `f3d6d238fc4e40a192ccb0f7b7ca594e39d86178696decf5ea134c8076c284ba` | no |
| `nonftv_phenotype_decodability_audit/metrics/residualizer_fits.csv` | `40640d2bd71fa21d377497e53ecb0ffaa579f921f61eaed6c8a3fdc66e2af020` | no |
| `nonftv_phenotype_decodability_audit/metrics/target_transform_fits.csv` | `22cd2b652ae6a943935e02c4426e1a7594403a4bcdeadce1dd49886bcde9a0f8` | no |
| `mri_clinical_complementarity_audit/reports/final_report.md` | `0236fd20d4d1170720b4bdc9e3af913a139be38bfcd711df316b9d21de142f3f` | no |

