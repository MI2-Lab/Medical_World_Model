# Clean-Code Validation

## Automated Tests

`pytest` currently covers:

- the paper 25-D condition and its causal prefix mask;
- DCE8 and 9-D geometry tensor construction;
- complete CoRe-JEPA forward, all selected pCR-free losses, and backward;
- train-fitted response-target normalization;
- the 1,283-D paper FLR contract;
- end-to-end FLR fitting and score/metric export.

## Real-Data Cache Equivalence

The clean cache builder was checked against the development cache used by the selected experiment.

| Cohort / patient | Artifact | Shape | Maximum absolute difference |
|---|---|---:|---:|
| I-SPY2 / `ACRIN-6698-102212` | DCE8 sequence | `[4,8,32,96,96]` | `0.0` |
| I-SPY1 / `ISPY1_1001` | DCE8 sequence | `[4,8,32,96,96]` | `0.0` |
| I-SPY2 / `ACRIN-6698-102212` | raw response features | `[4,106]` | `0.0` |
| I-SPY1 / `ISPY1_1001` | LD-based raw response features | `[4,106]` | `0.0` |

The NaN masks of both response-feature rows also matched exactly. These comparisons verify the less obvious cohort-specific details: I-SPY2 bbox-centered crops, I-SPY1 automatic ROI projection/fallback, adaptive DCE8 phases, BreastDCEDL/all-post response phases, and I-SPY1 LD-based guidance.

Across all 964 mixed-cohort records, the transformed 18-D response vector matched the development target exactly (`max_abs_diff=0`). The treatment-family-standardized scalar response score agreed to floating-point roundoff (`max_abs_diff=3.73e-9`).

## Split and Condition Audit

On the shared processed data, the clean loader finds:

```text
808 I-SPY2 primary records
156 I-SPY1 pretraining-only records
565 primary training records
721 total pCR-free pretraining records
121 validation records
122 locked-test records, including 42 pCR-positive
14 exact treatment arms
25 condition dimensions
```

## Two-GPU Check

`python scripts/smoke_model.py --gpus 0,1 --batch-size 4` completed a synthetic forward, complete loss, and backward with `data_parallel=True`. The response guidance heads are evaluated inside the model forward, so both devices receive their local image and response-state work. This removes the development script's redundant second response-state pass on the primary GPU while preserving the stated objective and tensor semantics.

A second two-GPU check used the real cached trajectories `ACRIN-6698-102212` and `ISPY1_1001` with the full paper dimensions. The complete forward/loss/backward consumed image `[2,4,8,32,96,96]` and condition `[2,3,25]`, and returned finite prediction `[2,3,192]` and future response state `[2,3,64]`.

## Scope

The clean module names and checkpoint schema differ from the experimental monolith. Input tensors and raw response targets have direct equivalence checks; architecture and active loss terms match the selected configuration. Because the clean trainer removes redundant stochastic recomputation and retraining is stochastic, it is not intended to produce a bitwise-identical checkpoint from the same seed.
