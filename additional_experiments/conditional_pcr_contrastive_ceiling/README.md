# Conditional pCR contrastive ceiling

This directory implements Goal B as an explicitly outcome-supervised ceiling
experiment. It is isolated from all prior experiment directories.

> This experiment intentionally uses pCR supervision and estimates a supervised representation ceiling. It is not evidence that the pCR-free World Model learned this information.

## Reproduction order

Use the PyTorch environment and point the input resolver at the private artifact
repository when it is not at the frozen default location:

```bash
export MWM_PRIVATE_INPUT_REPO_ROOT=/data/mi2-interns/bowen/Medical_World_Model
export PYTHONPATH="$PWD/additional_experiments/conditional_pcr_contrastive_ceiling/src"
PYTHON=/home/bowen/.conda/envs/bowen/bin/python

$PYTHON additional_experiments/conditional_pcr_contrastive_ceiling/scripts/run_matrix.py \
  --arms B1,B2,B3 --seeds 2026,3026 --folds 0,1,2,3,4 \
  --gpus 0,1,2 --workers 3 --skip-complete --execute
$PYTHON additional_experiments/conditional_pcr_contrastive_ceiling/scripts/run_evaluation.py
$PYTHON additional_experiments/conditional_pcr_contrastive_ceiling/scripts/generate_figures.py
$PYTHON additional_experiments/conditional_pcr_contrastive_ceiling/scripts/generate_report.py
$PYTHON additional_experiments/conditional_pcr_contrastive_ceiling/scripts/verify_experiment.py
```

`run_matrix.py` is a dry-run preflight unless `--execute` is supplied. It
validates complete cells before resuming. Patient-level representations,
checkpoints, predictions, per-cell matching records, and logs have
`.private.*` names or live below fully ignored artifact trees. Only aggregate
metrics, figures, source, tests, locks, and reports are eligible for Git.

The primary analysis uses T0, T0–T1, and T0–T2. T0–T3 is generated only as a
supplementary estimate and cannot enter Gates A–C. The 808-patient and
375-patient FTV estimands are evaluated separately.
