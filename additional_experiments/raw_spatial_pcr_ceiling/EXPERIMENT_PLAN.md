# Goal C — Raw-Image / Spatial pCR Ceiling Audit

## Scientific question and boundary

This is a supervised oracle/ceiling diagnostic. It deliberately uses pCR during
training to estimate the **empirical ceiling under the current C1B-H input
contract**. It is not a pCR-free World Model, treatment-causal evidence,
external validation, production model, or information-theoretic ceiling.

The primary question is whether exposing the full pre-pooling spatial map,
learning a small supervised spatial readout, or optimizing the existing 3-D
encoder directly can materially exceed the previous pooled LOCAL supervised
ceiling (approximately AUROC 0.55; best prior supervised MRI-only result
approximately 0.548).

## Required prior evidence

The following prior reports were read before this plan was frozen:

1. `conditional_pcr_contrastive_ceiling`: the strongest supervised LOCAL
   reference remained low (best MRI-only AUROC about 0.548), with negligible
   ceiling gap and a clear train/test gap; full encoder fine-tuning did not
   materially raise the ceiling.
2. `local_response_state_multiseed_confirmation`: LOCAL0 and LOCAL3 were
   stable across seeds; LOCAL3 is the confirmed pooled response baseline.
3. `spatial_heterogeneity_phenotype_audit`: spatially localized phenotype
   information was suggested by the diagnostic oracle, but fixed pooling
   variants did not establish a pCR gain or FTV-independent value.
4. `mask_free_region_aware_audit`: fixed central/inner/outer shells did not
   recover a stable pCR or beyond-FTV signal; the result was diagnostic and
   not a biological localization claim.
5. `mri_clinical_complementarity_audit`: the current MRI state had weak pCR
   discrimination and no stable clinical or clinical+FTV incremental value.
6. `foundation_mri_baselines`: MedicalNet and DINOv1 references gave mixed
   results and did not establish a consistent beyond-FTV gain or justify
   immediate encoder replacement.
7. `foundation_mri_dinov3_posthoc`: the post-hoc DINOv3 comparison was
   diagnostic, not a clean external validation, and did not remove the need
   for this controlled supervised ceiling test.

Together these establish the required next diagnostic: supervised spatial
learning must test whether useful pCR information was lost specifically by
pooling, rather than infer that from fixed region probes.

## Frozen population, timing, and replication

- `full_808` is used for MRI-only, clinical-only, and clinical+MRI.
- `ftv_complete_375` is used only for FTV and clinical+FTV+MRI analyses.
- The exact prior five outer folds are reused; seeds are `2026` and `3026`.
- Headline timings are `T0`, `T0–T1`, and `T0–T2`.
- `T0–T3` is supplementary and always labeled `late/pre-surgery`.
- C1B-H DCE7, physical geometry, preprocessing, visit timing, and technical
  eligibility are unchanged. FOV is not expanded and no BPE source proxy is
  used.

## Arms

| Arm | Input/readout | Trainable parameters | Purpose |
|---|---|---|---|
| C0 | previous pooled supervised LOCAL3 reference; original B0 LOCAL3 also reported | reused artifact | pooled ceiling reference |
| C1 | full final encoder map → valid-map GAP → small BCE head | head only; optional final-stage light fine-tune | expose spatial field without learned localization |
| C2 | 64-mm LOCAL tokens → one-query cross-attention (≤2 blocks, 4 heads) | aggregator/head; optional final-stage light fine-tune | supervised localization within LOCAL |
| C3 | 64-mm LOCAL tokens → 3-block, 4-head token Transformer | aggregator/head; optional final-stage light fine-tune | strongest supervised LOCAL spatial ceiling |
| C4 | all full-map tokens → C3-matched Transformer | aggregator/head; optional final-stage light fine-tune | test context outside LOCAL |
| C5 | raw C1B-H DCE7 → existing 3-D encoder → C4-matched spatial readout | full supervised encoder adaptation | test representation bottleneck |
| C6 | frozen DINOv3 LOCAL token features → low-capacity head | secondary, disabled unless requested | architecture-family sensitivity |

Clinical variables never enter the MRI extractor or spatial aggregator. They
enter only fold-safe downstream fusion models `C`, `C+M`, `C+F`, and `C+F+M`.
Primary loss is `BCEWithLogitsLoss` with train-fold class weighting when
configured. No contrastive loss, treatment matching, triplet loss, or clinical
conditioned negative sampling is used.

## Selection and evaluation

Architecture and primary hyperparameters are frozen by
`PREREGISTRATION_LOCK.json`; validation may select only from the small
registered LR/weight-decay grid and early stopping. The outer test is never
used for selection. Every arm exports train, validation, and outer-test/OOF
AUROC, AUPRC, Brier, calibration slope, ECE, and the train−OOF gap.

Attention diagnostics (entropy, concentration, center/outer mass,
longitudinal and seed consistency) are descriptive only and cannot select an
architecture. Paired bootstrap uses at least 5,000 patient-level draws within
outer fold on the same patients; folds are not independent biological
replicates.

## Decision gates

- Gate A: C2 or C3 exceeds C0 by mean ΔAUROC ≥ +0.05 at T0–T1 or T0–T2 and
  both seeds are positive.
- Gate B: C5 exceeds C0 by mean ΔAUROC ≥ +0.08 at an early/mid timing and
  both seeds are positive.
- Gate C: C4 exceeds C3 by mean ΔAUROC ≥ +0.03 and both seeds agree.
- Gate D: the best spatial arm has positive `C+M−C` in both seeds at an
  early/mid timing; strong form is mean ΔAUROC ≥ +0.03.
- Gate E: on matched 375, `C+F+M−(C+F)` is positive in both seeds; strong
  form is mean ΔAUROC ≥ +0.03.

If train AUROC exceeds 0.80 while OOF remains near 0.55, the result is
`SUPERVISED_SPATIAL_OVERFIT`, not `SPATIAL_SIGNAL_FOUND`.

## Privacy and delivery

Patient identifiers, labels, raw MRI, checkpoints, patient-level features,
and predictions remain private and are gitignored. Public artifacts contain
only aggregate metrics, hashes, contracts, and the Chinese final report.
Old experiment directories are read-only. Delivery requires a clean audit of
this new subtree, an old-tree immutability check, the requested commit, and a
non-force push attempt. If the push fails the report records
`GITHUB_PUSH_FAILED`.

