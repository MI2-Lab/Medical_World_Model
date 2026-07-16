# Tensor Contracts

## Symbols

| Symbol | Meaning | General shape | Paper-v1 shape |
|---|---|---:|---:|
| `B` | mini-batch size | scalar | 32 globally |
| `V` | observed trajectory visits | variable | 4 |
| `L` | valid next-visit transitions | `V-1` | 3 |
| `C_x` | DCE visit channels | configurable | 8 |
| `D_q` | lesion descriptor dimension | configurable | 9 |
| `C_c` | condition dimension | cohort-dependent | 25 |
| `D_z` | visit latent dimension | configurable | 192 |
| `D_s` | future response-state dimension | configurable | 64 |
| `D_r` | response-guidance vector dimension | configurable | 18 |
| `E` | response-state experts | configurable | 6 |

Spatial crop sizes are always written as `(Z,Y,X)`. NIfTI arrays are loaded as `(X,Y,Z,T)` and converted explicitly.

## Patient Cache

Each `<tensor_cache>/<patient_id>.npz` contains:

| Key | Meaning | Shape / dtype |
|---|---|---|
| `image` | four DCE8 visits | `[4,8,Z,Y,X]`, `float32` |
| `geometry` | `q_0,...,q_3` | `[4,9]`, `float32` |
| `phase_indices` | pre/early/late source frames | `[4,3]`, `int16` |
| `roi_sources` | ROI provenance per visit | `[4]`, string |
| `channel_names` | DCE8 channel semantics | `[8]`, string |

The eight channels are:

1. precontrast intensity;
2. selected early postcontrast intensity;
3. selected late postcontrast intensity;
4. early minus pre;
5. late minus pre;
6. peak relative enhancement over all postcontrast frames;
7. late-minus-peak relative enhancement (washout);
8. binary localization ROI.

The first seven channels are robust-normalized independently per visit. The ROI channel is not normalized.

## Lesion Descriptor `q_t`

`q_t` is computed from the final cropped ROI supplied to the model:

| Index | Feature |
|---:|---|
| 0 | ROI volume fraction in the crop |
| 1-3 | normalized bounding-box size in Z, Y, X |
| 4 | bounding-box volume fraction |
| 5 | ROI fill fraction inside its bounding box |
| 6-8 | normalized center in Z, Y, X, mapped to `[-1,1]` |

`q_t` is not a manual radiologist annotation. For I-SPY2 it is usually derived from the nonempty FTV inclusion region; the released analysis mask itself is a bit-encoded exclusion map. For I-SPY1 or missing regions, the cache records the fallback source. The paper configuration labels the rare development-compatible empty-FTV/full-field case as `legacy_full_field_empty_ftv`, so it is visible rather than silently treated as tumor segmentation.

## Condition `c_t`

The condition encoder returns `[B,3,C_c]`. Each row corresponds to one target visit:

```text
3 target-visit one-hot entries
4 observed-prefix mask entries
N_arm exact treatment-arm one-hot entries
3 biomarker entries: HR, HER2, MammaPrint
1 standardized age entry
```

For the mixed paper cohort, `N_arm=14`, hence `C_c=3+4+14+3+1=25`. The first seven entries are temporal. Expert routing removes those seven entries and sees only treatment and baseline patient context.

## Model Flow

### Observed visit representation

```text
image encoder:       [B*4,8,Z,Y,X] -> [B*4,D_z]
JEPA projector:      [B*4,D_z]     -> [B*4,D_z]
geometry projector: [B*4,9]       -> [B*4,D_z]
visit state z:       appearance + geometry -> [B,4,D_z]
```

### Image transition

```text
inputs:  z_0:z_2 [B,3,D_z], condition [B,3,C_c]
output:  z_hat_image_1:3 [B,3,D_z]
```

The causal output at row 0 sees `z_0`; row 1 sees `z_0,z_1`; row 2 sees `z_0,z_1,z_2`.

### Future response state

```text
inputs:  q_0:q_2 [B,3,9], condition [B,3,C_c]
outputs: s_hat_1:3 [B,3,D_s]
         q_hat_1:3 [B,3,9]
         delta_z_response [B,3,D_z]
         gate probabilities [B,3,E]
```

There is no separately encoded observed `s_t` in paper v1. A prefix ending at `t` directly forecasts `s_hat_(t+1)`. `q_hat` is an internal decoded geometry code used by the latent adapter. Its direct regression weight is zero, and it must not be interpreted as a predicted segmentation.

### Joint prediction

```text
z_hat_(t+1) = z_hat_image_(t+1) + delta_z_response_(t+1)
target: EMA-encoded observed next visit [B,3,D_z]
```

The target state includes the target visit's image and `q_(t+1)`. Target-module parameters are updated by EMA with momentum 0.996.

## pCR-Free Guidance

The raw response cache stores `[N,4,106]` per-visit shape and DCE statistics. I-SPY2 rows are extracted from the ROI and original DCE curve. In the selected mixed-cohort protocol, I-SPY1 rows use released longitudinal longest diameter, with `LD^3` as a volume proxy and kinetic entries left missing for train-split imputation. The train-split transform produces:

| Batch key | Meaning | Shape |
|---|---|---:|
| `response_vector` | standardized 18-D next-visit kinetic/shape response | `[B,3,18]` |
| `response_score` | treatment-family-standardized scalar imaging response | `[B,3,1]` |
| `routing_target` | six-way treatment-family/cohort stratum | `[B]` |

The 18-D vector contains baseline-relative and previous-visit-relative values for ROI volume, bbox volume, longest bbox diameter, bbox fill, early enhancement, peak enhancement, washout, enhancement AUC, and time to peak.

The pretraining dataset intentionally has no pCR key.

## Frozen Landmark Readout

At landmark `k`, available future states are `s_hat_1,...,s_hat_(k+1)`. FLR concatenates:

```text
first state
current state
prefix mean
most recent state revision
displacement from first state
```

This base has `5*D_s` entries. Landmark interactions contribute `3*5*D_s`, and the landmark one-hot contributes 3:

```text
FLR dimension = 20*D_s + 3 = 1283 when D_s=64.
```

One class-balanced logistic model is fit to all three stacked training landmarks. pCR first enters the pipeline here.
