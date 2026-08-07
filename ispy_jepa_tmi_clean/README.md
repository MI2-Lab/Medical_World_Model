# CoRe-JEPA for Longitudinal I-SPY DCE-MRI

This directory is the standalone, paper-oriented implementation of the I-SPY
longitudinal world model. It contains the full path from raw DICOM conversion to
pCR-free CoRe-JEPA pretraining and frozen pCR evaluation. It does not import any
module from the original experimental development tree.

## What Is Included

- Raw I-SPY2 and I-SPY1 DICOM conversion and manifest generation.
- I-SPY2 analysis-mask decoding, clinical-label extraction, and DCE timing audit.
- Auditable four-visit DCE8 tensor and 9-D lesion-descriptor construction.
- Raw pCR-free kinetic/shape response-feature extraction and train-split normalization.
- Modular 3-D visit encoder, conditioned image transition, factorized future response state, EMA target encoder, and selected IRG objective.
- Frozen future-state export and the shared Frozen Landmark Readout (FLR).
- Tensor-contract documentation, a portable paper configuration, unit tests, a real-data cache smoke path, and a two-GPU model smoke test.

The clean implementation mirrors the selected paper-v1 architecture and loss configuration. Module names were deliberately simplified, so legacy development checkpoints are not loaded by name; retraining this package creates clean checkpoints with explicit configuration and preprocessing metadata.

## Layout

```text
ispy_jepa_tmi_clean/
  configs/paper_v1.yaml              paper architecture and training settings
  corejepa/
    data/                             records, DCE8, q, conditions, response targets
    models/                           encoder, conditioned transitions, CoRe-JEPA
    training/                         pCR-free losses, training, checkpoint export
    readout/                          shared frozen landmark logistic readout
  data_processing/
    preprocessing/                    DICOM/NIfTI and clinical preprocessing scripts
    metadata/                         portable BreastDCEDL phase metadata
    scripts/run_data_processing.py    raw-data stage runner
  scripts/
    build_tensor_cache.py
    build_response_cache.py
    pretrain.py
    fit_readout.py
    smoke_model.py
  docs/
  tests/
```

## Installation

From this directory:

```bash
conda create -n corejepa python=3.11 -y
conda activate corejepa
pip install -e ".[excel,test]"
```

`dcm2niix` is an external executable and is needed only when rebuilding NIfTI files from DICOM. PyTorch must be installed with a CUDA build appropriate for the machine when GPU training is required.

## Shared Data On This Server

The already processed data are available without any `/home/<user>` dependency:

```text
/data/data/Preprocessed/I-SPY2
/data/data/Preprocessed/I-SPY1
```

The portable configuration already points to these locations. Raw-data paths and `dcm2niix` are configured separately in `data_processing/config/paths.shared-data.env` or a machine-local env file.

## End-to-End Commands

Run all commands from this directory.

1. Check raw and processed paths:

```bash
python data_processing/scripts/run_data_processing.py \
  --env-file data_processing/config/paths.shared-data.env \
  --stage check
```

2. Rebuild DICOM-derived NIfTI/manifests when needed:

```bash
python data_processing/scripts/run_data_processing.py \
  --env-file data_processing/config/paths.local.env \
  --stage all --execute
```

3. Build the model-ready DCE8 and `q_t` caches:

```bash
python scripts/build_tensor_cache.py --config configs/paper_v1.yaml
```

4. Build raw pCR-free response descriptors:

```bash
python scripts/build_response_cache.py --config configs/paper_v1.yaml
```

5. Pretrain CoRe-JEPA. This stage never places pCR in a training batch:

```bash
python scripts/pretrain.py --config configs/paper_v1.yaml
```

6. Fit and evaluate FLR after the representation is frozen:

```bash
python scripts/fit_readout.py --config configs/paper_v1.yaml
```

## Smoke Checks

```bash
pytest
python scripts/smoke_model.py
python scripts/smoke_model.py --gpus 0,1
python scripts/build_tensor_cache.py \
  --config configs/paper_v1.yaml \
  --cache-dir /tmp/corejepa_tensor_smoke --limit 1 --overwrite
```

## Important Representation Boundary

The model learns two connected pathways. The image pathway uses DCE8 visit states to predict the next visit latent. The factorized response pathway uses the observed lesion descriptors `q_0:t` together with clinical/treatment condition to produce a future response state `s_hat_(t+1)`; that state corrects the image-driven next-latent forecast. The primary FLR uses the frozen future response states. It does not append clinical variables again, and it does not append the image latent to the primary readout.

This is intentional and explicit in the code. See [Tensor Contracts](docs/TENSOR_CONTRACTS.md), [Pipeline](docs/PIPELINE.md), [Module I/O](docs/MODULE_IO.md), and [Validation](docs/VALIDATION.md).
