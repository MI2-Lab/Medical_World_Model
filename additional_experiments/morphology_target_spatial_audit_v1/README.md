# Morphology Target 与 Spatial Information Audit V1

This is an immutable, outcome-blind audit of the V6 morphology failure. It
reuses the V6 DINO spatial cache and fixed C0/folds, but writes all new targets,
probes and pilot outputs under this independent directory. It does not read
pCR, clinical outcomes, or outcome-derived metrics.

Run from this directory with Anaconda `bowen`:

```bash
conda run -n bowen python scripts/run_phase_a.py
conda run -n bowen python scripts/run_phase_b.py
```

Phase B is conditional on the Phase A target-quality gate. If no morphology
candidate is eligible, it exits without creating pilot checkpoints.
