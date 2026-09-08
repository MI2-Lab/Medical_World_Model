# MAMA-MIA-inspired T0 static DCE baseline

This experiment trains two lesion-centred, T0-only pCR classifiers on the
repository's locked seed-2026 patient folds. It deliberately rejects every
path containing `T1`, `T2`, or `T3`. DCE paths are resolved from each T0
manifest so repaired partial-z acquisitions use the declared aligned NIfTI.

```bash
export ISPY2_PREPROCESSED_ROOT=/path/to/I-SPY2
PYTHON=/path/to/python-with-torch
$PYTHON scripts/run_experiment.py audit
$PYTHON scripts/run_experiment.py build-cache
$PYTHON scripts/run_experiment.py train --variant subtraction --fold 0 --device cuda:0
$PYTHON scripts/run_experiment.py train --variant three_phase --fold 0 --device cuda:1
$PYTHON scripts/run_experiment.py evaluate
$PYTHON scripts/run_experiment.py zero-image --device cuda:0
$PYTHON scripts/run_experiment.py train --variant three_phase --fold 0 \
  --device cuda:0 --shuffled-labels --tag random_label
```

Repeat training for folds 0--4. Patient-level manifests and predictions are
kept local by `.gitignore`; aggregate metrics and the Chinese report are safe
to commit.
