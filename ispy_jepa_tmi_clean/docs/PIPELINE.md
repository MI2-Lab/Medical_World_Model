# End-to-End Pipeline

## 1. Raw DICOM and Clinical Data

The portable raw-data runner is `data_processing/scripts/run_data_processing.py`. Paths are supplied by an env file or environment variables. No `/home/<user>` path is required for the already processed data.

Raw conversion produces one patient manifest and one full 4-D DCE NIfTI per visit. I-SPY2 additionally stores the raw analysis mask and the derived binary FTV inclusion region. The manifest is the authoritative connection between raw series, converted volumes, phase count, mask, and bounding box.

The I-SPY2 analysis mask must not be described as a manual dense tumor segmentation. Zero-valued voxels represent the FTV inclusion region; nonzero values encode exclusion conditions used by the source analysis workflow.

## 2. Model Tensor Cache

`scripts/build_tensor_cache.py` performs these operations for every trajectory:

1. Read and spatially canonicalize all four DCE NIfTIs.
2. Load a nonempty FTV inclusion region, otherwise a released bbox, otherwise an automatic enhancement ROI. `paper_v1.yaml` explicitly retains the development pipeline's rare empty-FTV/full-field compatibility behavior; setting `legacy_empty_ftv_full_field: false` applies the strict nonempty fallback instead.
3. Use the released T0 bbox center for I-SPY2 and the automatic-mask centroid for I-SPY1, then project that center by normalized coordinates to follow-up visits. An automatic ROI can recenter if the projected crop captures fewer than 32 voxels or less than the configured fraction.
4. Select pre/early/late phases using the included BreastDCEDL metadata. Series with four or fewer frames use first-post and final-frame fallback. Peak enhancement always considers every postcontrast frame.
5. Construct and robust-normalize seven image channels, then append the binary ROI.
6. Compute `q_t` from that exact cropped ROI.
7. Save tensors, selected phase indices, channel names, and ROI provenance.

This cache is independent of pCR.

## 3. Response-Guidance Cache

`scripts/build_response_cache.py` computes raw per-visit ROI geometry and enhancement statistics from the original I-SPY2 DCE arrays. It saves 106 values per visit. To reproduce the selected mixed-cohort experiment, I-SPY1 guidance rows use the four released longest-diameter measurements: the three bbox dimensions are set to LD, volume fields use `LD^3`, and unavailable kinetic fields remain missing until train-split imputation. This does not affect the I-SPY1 DCE8 image tensor or its automatic localization ROI. The raw cache is not normalized and can be audited independently.

At training setup, only the pretraining partition is used to fit:

- missing-value medians;
- means and standard deviations for the 18-D response vector;
- 2nd/98th percentile clipping and standardization for the six score components;
- treatment-family/cohort score means and standard deviations.

The transformed targets describe imaging response and do not use pCR.

## 4. Patient Split and External Pretraining Records

I-SPY2 is split at patient level using `split_seed=2026` into 70% train, 15% validation, and 15% locked test. I-SPY1 records are appended only to the pCR-free pretraining partition. They are not appended to validation, test, or FLR fitting.

The exact-arm vocabulary is established once from the mixed record set. With the paper cohort it has 14 entries. Routing uses six broader treatment-family/cohort strata.

## 5. CoRe-JEPA Pretraining

The image path predicts three targets in one causal pass:

```text
T0 -> T1 latent
T0,T1 -> T2 latent
T0,T1,T2 -> T3 latent
```

The response path follows the same prefixes using `q` and condition. It predicts future response states and contributes a learned correction to each image-driven latent forecast.

The optimizer sees no pCR. The best checkpoint is selected by validation next-latent prediction loss, with a minimum latent standard deviation used to reject collapsed states.

## 6. Frozen State Export and FLR

After loading the selected checkpoint, `forecast_response` exports three future response states for every patient. pCR labels are then loaded for the primary I-SPY2 records only. One shared logistic readout is trained across the T0, T0+T1, and T0+T1+T2 landmarks. Hyperparameters are selected on validation AUROC using weights 2/1/0.5 and evaluated once on the locked test set.

## Output Directory

The default `runs/corejepa_clean` contains:

```text
config.yaml
history.csv
best_corejepa.pt
last_corejepa.pt
frozen_states.npz
splits.json
flr.pkl
flr_metrics.csv
flr_scores.csv
flr_summary.json
```

The checkpoint also stores the condition vocabulary, age statistics, response-target transform, patient order, and split indices.
