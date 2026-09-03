# DINOv3 Spatial Parallel Radiomics Adapter V6

V6 tests whether preserving DINOv3 patch positions enables an independent
radiomics branch to learn morphology and texture beyond the frozen V5 C0
state.  The representation stage is outcome-blind; pCR remains locked until
the formal mechanism lock is created.

Run with Anaconda `bowen`:

```bash
conda run -n bowen python scripts/build_targets.py
conda run -n bowen python scripts/preflight.py
# only after target_feasibility.json is PASS:
CUDA_VISIBLE_DEVICES=0 python scripts/extract_spatial.py --device cuda --num-shards 3 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python scripts/extract_spatial.py --device cuda --num-shards 3 --shard-index 1
CUDA_VISIBLE_DEVICES=2 python scripts/extract_spatial.py --device cuda --num-shards 3 --shard-index 2
```

V2-V5 are immutable parents. Private caches, targets, states and checkpoints
are ignored and are represented by SHA manifests only.

## Pilot status

The 2026 pilot completed all 10 cells (five folds × S0/S1). The spatial arm
passed the overall representation gates (`direct-head=0.4708`, matched probe
`0.3118`, S1−S0=`+0.0468`) and texture/intensity/kinetic family checks. It
failed the preregistered morphology-specific spatial gain gate (S1−S0=`+0.0010`,
required `+0.03`). The decision is `NO_GO`; formal fresh-seed training and pCR
evaluation remain locked. See `reports/final_report.md` and
`metrics/pilot_gate.json`.
