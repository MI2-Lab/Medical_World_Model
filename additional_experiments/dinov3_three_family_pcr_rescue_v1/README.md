# DINOv3 Three-Family Radiomics → pCR Rescue v1

This experiment removes morphology from the critical path. It reuses the immutable V6 DINO cache, V6 three-family transfer evidence, and V6 seed-2026 states for a locked exploratory conditional pCR evaluation. Fresh-seed formal training is started only if the exploratory direction rule passes.

All representation training is pCR-blind. The pCR evaluator opens outcome data only after `EXPLORATORY_EVALUATION_LOCK.json` (Stage A) or both `EVALUATION_LOCK.json` and `MECHANISM_LOCK.json` (formal Stage C) exist and validate.

Run in the `bowen` conda environment:

```bash
conda run -n bowen python scripts/run_stage_a_lock.py
conda run -n bowen python scripts/evaluate_exploratory_pcr.py
conda run -n bowen python scripts/decide_exploratory.py
```

Fresh-seed formal training is intentionally not launched by the exploratory command. It is authorized only by `EXPLORATORY_CONTINUE.json`.
