# DINOv3 Parallel Radiomics Adapter V5

V5 is the causal isolation test following the V4 shared-gradient diagnostic.
The C0 JEPA response is frozen and a cloned, independent MRI adapter learns
only residual-radiomics PCA16 targets.  The exported primary state is
`[C0 192-D, radiomics 64-D] = 256-D`; inference accepts only frozen DINO
summaries.  V2-V4 assets and decisions are immutable parents.

Run with Anaconda `bowen`:

```bash
conda run -n bowen python scripts/preflight.py
conda run -n bowen python scripts/run_isolation_smoke.py
conda run -n bowen python scripts/run_pilot.py --device cuda --num-shards 3 --shard-index 0
```

Pilot and formal representation stages are outcome-blind.  pCR remains locked
until both the formal evaluation lock and mechanism lock are valid.  Private
checkpoints, states, patient-level targets and manifests are ignored.
