# DINOv3 Factorized Radiomics Grounding V4

This experiment tests whether JEPA and residual-radiomics grounding can share a
frozen-DINO MRI adapter when the 192-D response state is factorized into a
128-D JEPA branch and a 64-D phenotype branch. V2 and V3 are immutable parents.

Representation training is outcome-blind. The pilot uses the hash-bound V2
DINO summaries and fold-train-only residualized PyRadiomics PCA16 targets; it
does not re-extract either asset. The pCR evaluator remains locked until the
formal representation matrix and outcome-blind mechanism lock are complete.

Run in Anaconda `bowen` after implementation:

```bash
conda run -n bowen python scripts/preflight.py
conda run -n bowen python scripts/run_head_only.py
conda run -n bowen python scripts/run_pilot.py --device cuda --num-shards 3 --shard-index 0
```

Generated checkpoints, states, patient-level data and private manifests are
ignored and must not be pushed to GitHub.
