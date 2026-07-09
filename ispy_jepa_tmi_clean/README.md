# I-SPY JEPA TMI Clean Code

This directory is the cleaned, paper-oriented code area for the I-SPY
longitudinal JEPA/world-model project.

The current cleaned module is:

- `data_processing/`: reproducible DICOM-to-NIfTI preprocessing, clinical-label
  extraction, MRI-NACT feature extraction, DCE phase audit, and data-path
  documentation.

The validated preprocessing scripts are copied into this clean tree under
`data_processing/preprocessing`. The code does not depend on the old development
folder layout.

## Repository Layout

```text
ispy_jepa_tmi_clean/
  README.md
  data_processing/
    README.md
    DATA_MANIFEST.md
    config/
      paths.example.env
      paths.shared-data.env
    preprocessing/
      *.py
    scripts/
      run_data_processing.py
```

## Quick Check

From the repository root:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py --stage check
```

On the current lab server, the processed I-SPY data live under `/data`. Use:

```bash
python3 ispy_jepa_tmi_clean/data_processing/scripts/run_data_processing.py \
  --env-file ispy_jepa_tmi_clean/data_processing/config/paths.shared-data.env \
  --stage check
```

For another machine, copy `data_processing/config/paths.example.env` to a local
env file, fill in that machine's data/tool paths, and pass it with `--env-file`.
