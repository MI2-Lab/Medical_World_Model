# Conditional pCR contrastive ceiling experiment

## Boundary

This is a deliberately outcome-supervised oracle/ceiling experiment. It asks whether C1B-H/DCE7 MRI contains pCR information conditional on HR, HER2, and assigned treatment that the current pCR-free representation has not organized. It is not a candidate pCR-free World Model and must never be reported as one.

## Frozen evidence and inputs

The protocol reads the completed evidence listed below without modifying any prior experiment directory.

| Evidence | Commit | Frozen report SHA-256 | Consequence for this experiment |
|---|---:|---|---|
| MRI–clinical complementarity audit | `064e059` | `0236fd20d4d1170720b4bdc9e3af913a139be38bfcd711df316b9d21de142f3f` | Existing LOCAL MRI-only pCR and incremental linear fusion are weak. |
| Compact MRI–clinical fusion audit | `20dfa4f` | `721ce68c7470147624b115a455694ae136f8940a65dc86773d5a3b58f7a17787` | Raw 192-D prefix concatenation overfits; use outer-train PCA with validation-selected 8/16/32/64 dimensions. |
| LOCAL multi-seed confirmation | `b4ec0c1` | `6f495ea2412b3fe6d6e24fa9c9f067f89685fa7e69c40758582f11d2855835da` | Use the confirmed fixed-64-mm LOCAL architecture and corresponding LOCAL3 checkpoints. |
| Classical DCE phenotype complementarity | `f49cf17` | `c9b1d5a6b4e7ea1cb9b8eb5002fcfe281b5fec36ae79fea7cf259356b69dfd0b` | Non-FTV handcrafted DCE phenotypes provide some complementary signal, motivating an information ceiling test. |
| Foundation MRI baselines | `98dfc5a` | `166250a9e259c02d8651906a4421c16311fd86b8db9c3f1bae9ce58ccaf13db0` | DINO and MedicalNet did not reliably solve pCR complementarity. |
| DINOv3 post-hoc baseline | `fbc31ff` | `2173d51d3f358c953f24d6a3584e96f1070477c543c73d16d0665744b8b1e2fb` | DINOv3 is not substituted for LOCAL3 in the primary ceiling. |

The full-cohort estimand is `full_808`; all FTV comparisons use the separate `ftv_complete_375` estimand. Frozen seed-2026 patient folds are reused exactly. Model seeds 2026 and 3026 are sensitivity replications over the same patients, not biological replicates.

## Conditional objective

For each outer fold, matching strata are constructed only from its training rows as exact `HR × HER2 × assigned arm`. A contrastive anchor is eligible only when the same training stratum contains at least one different patient with the same pCR label and at least one patient with the opposite pCR label. Positives and negatives never cross strata; test patients and unmatched fallback negatives are forbidden.

`B1` minimizes conditional supervised contrastive loss only. `B2` and `B3` minimize:

`L = L_condSupCon + 0.25 L_pCR`.

Clinical fields determine matching only. They are never passed into the MRI encoder, projection head, or pCR logits.

For B2/B3, every stochastic epoch visits every eligible anchor exactly once,
with a deterministic epoch-specific reshuffle. Each logical batch contains
three or four unique patients from one exact stratum: the anchor, a different
same-pCR partner, an opposite-pCR partner, and, when the stratum permits it, one
additional unique member. Four is a configured maximum, not a promise that a
fourth unique member exists in every 2-v-1 stratum. Encoder microbatching may
reduce device memory without changing this sampling budget.

## Arms and longitudinal contract

- `B0`: confirmed LOCAL3 response state, frozen and outcome-free.
- `B1`: frozen LOCAL3; train only a 192→128→64 nonlinear contrastive projection.
- `B2`: start from LOCAL3; train encoder stage 4, LOCAL response projection, the small contrastive projection, and training-only linear logits.
- `B3`: start from LOCAL3; train the full image encoder at low learning rate plus the same small heads. This arm is diagnostic only.

At T0, T0–T1, T0–T2, and supplementary T0–T3, only observed visit representations are concatenated. A PCA fitted on outer train only produces 8/16/32/64-D candidates; dimension and logistic regularization are chosen from validation AUROC before test evaluation. No pCR Transformer or nonlinear clinical fusion head is allowed.

## Evaluation and public artifacts

After supervised training, the MRI representation is frozen. Fold-isolated regularized logistic regression evaluates `M`, `C`, `C+M`, and, only within the 375-patient population, `F`, `C+F`, and `C+F+M`. Metrics are AUROC, AUPRC, Brier score, calibration slope, and 10-bin ECE. Headline deltas use 5,000 paired patient bootstrap draws stratified by outer fold.

Public Git artifacts contain aggregate metrics, figures, source code, protocol locks, verification, and the Chinese report. Checkpoints, feature matrices, patient-level predictions, bootstrap draws, and patient manifests remain private/ignored.
