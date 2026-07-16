# Module Input and Output Reference

## Data Modules

### `corejepa.data.nifti`

- `read_dce_nifti(path)`
  - Input: uncompressed `.nii` path.
  - Output: DCE array `[X,Y,Z,T]` and metadata dictionary.
  - I-SPY1 slice-first volumes are conservatively canonicalized to XYZ.
- `read_spatial_nifti(path)`
  - Input: spatial `.nii` path.
  - Output: image/mask `[X,Y,Z]` and metadata.

### `corejepa.data.imaging`

- `select_phase_indices(n_phases, metadata, policy)`
  - Output: `(pre, early, late, peak_window)` integer indices.
- `load_visit_roi(visit, dce, fallback)`
  - Input: manifest visit dictionary and DCE `[X,Y,Z,T]`.
  - Output: binary ROI `[X,Y,Z]` plus provenance string.
- `dce8_visit(...)`
  - Inputs: DCE `[X,Y,Z,T]`, ROI `[X,Y,Z]`, XYZ crop center.
  - Outputs: DCE8 `[8,Z,Y,X]` and three source phase indices.
- `mask_geometry(mask)`
  - Input: cropped ROI `[Z,Y,X]`.
  - Output: `q_t [9]`.
- `build_patient_tensor(...)`
  - Input: one four-visit `PatientRecord`.
  - Output: one auditable `.npz` patient cache.

### `corejepa.data.condition.ConditionEncoder`

- Constructor input: all records used to establish a stable exact-arm vocabulary and age statistics.
- `encode(record)` output: transition condition `[3,C_c]`.
- `routing_target(record)` output: scalar class in `[0,5]`.

### `corejepa.data.response_targets`

- `build_response_feature_cache(records, output)`
  - Output: raw descriptors `x_visit [N,4,106]`.
- `response_vector(x_visit, names)`
  - Output: raw next-visit changes `[N,3,18]`.
- `ResponseTargetTransform.fit(raw, records, train_indices)`
  - Fits imputation, clipping, scaling, and family standardization only from the pretraining partition.
- `transform(raw, records)`
  - Outputs standardized vector `[N,3,18]` and score `[N,3,1]`.

### `corejepa.data.dataset`

- `LongitudinalDCEDataset[index]` returns image, geometry, condition, routing target, record index, and patient ID.
- `PretrainingDataset[index]` adds response vector and score but deliberately does not return pCR.

## Model Modules

### `VisitEncoder3D`

- Input: `[B,8,Z,Y,X]`.
- Output: `[B,D_z]`.
- Four residual stages use widths 16/32/64/128 in paper v1.

### `ConditionedCausalTransformer`

- Inputs: state sequence `[B,L,D]`, condition `[B,L,C_c]`.
- Output: prefix-causal hidden sequence `[B,L,D]`.
- Condition enters through an additive projection and FiLM.

### `ImageTransition`

- Inputs: observed visit states `[B,3,D_z]`, condition `[B,3,C_c]`.
- Output: image-driven next states `[B,3,D_z]`.

### `FutureResponseState`

- Inputs: observed descriptor rows `[B,3,9]`, condition `[B,3,C_c]`.
- Outputs are grouped in `ResponseStateOutput`:
  - `future_state [B,3,D_s]`;
  - `decoded_geometry [B,3,9]`;
  - `latent_correction [B,3,D_z]`;
  - `gate_logits` and `gate_probabilities [B,3,E]`.

### `CoReJEPA`

- Inputs: `image [B,4,8,Z,Y,X]`, `geometry [B,4,9]`, `condition [B,3,C_c]`.
- Output: named `CoReJEPAOutput`; every field and shape is documented in `corejepa/models/corejepa.py`.
- `forecast_response(geometry, condition)` returns exactly the frozen `[B,3,D_s]` representation used by FLR.

## Training and Readout Modules

### `PretrainingObjective`

- Inputs: `CoReJEPAOutput` and pCR-free batch fields.
- Output: scalar differentiable loss and flat scalar-statistics dictionary.
- Active terms are next-latent prediction, SIGReg, response score/rank, state-delta contrast, response update, 18-D response vector/update, and gate route/entropy/balance.
- No direct geometry regression, pCR auxiliary loss, endpoint supervision, or checkpoint-by-AUC rule is present.

### `training.runner.train`

- Input: `ExperimentConfig`.
- Outputs: `best_corejepa.pt`, `last_corejepa.pt`, `history.csv`, `frozen_states.npz`, and `splits.json`.
- Checkpoint selection uses validation next-latent prediction loss subject to the latent-dispersion threshold.

### `readout.flr.landmark_features`

- Input: future states `[N,3,D_s]` and landmark `0/1/2`.
- Output: `[N,20*D_s+3]`.

### `fit_frozen_landmark_readout`

- Inputs: frozen state file, locked split file, and readout config.
- Outputs: `flr.pkl`, `flr_metrics.csv`, `flr_scores.csv`, and `flr_summary.json`.
