# DINOv3 MRI adapter + direct radiomics grounding V2

This experiment asks whether a trainable MRI-domain adapter over frozen DINOv3
slice summaries can produce pCR-relevant image information beyond clinical data
and FTV. V2 keeps the 64-voxel target ROI, uses valid-source-bounded morphology,
and grounds T0--T2 only. Representation learning is outcome-blind. The pCR
evaluator remains fail-closed until all 75 D1/D2/D3 cells and all
`[808,4,192]` state files have been frozen into `EVALUATION_LOCK.json` and the
outcome-blind mechanism gate has created `MECHANISM_LOCK.json`.

## Frozen result

V2 passed ROI feasibility, all five radiomics-target folds, the 947-patient
DINO cache contract, and the paired smoke cell. The formal matrix stopped at
`seed4026_fold4_D2`: after all 12 epochs, its best validation JEPA loss was
0.06056 versus the paired-D1 safety ceiling of 0.05293. The frozen decision is
`GROUNDING_OPTIMIZATION_CONFLICT`; 49/75 complete state cells existed at the
hard stop. Neither mechanism evaluation nor pCR evaluation ran, and no pCR
outcome was read. See [`reports/final_report.md`](reports/final_report.md).

Main commands use the Anaconda `bowen` environment:

```bash
conda run -n bowen python scripts/preflight.py
conda run -n bowen python scripts/build_radiomics_rois.py
conda run -n bowen python scripts/extract_radiomics.py --radiomics-python <py39-python>
conda run -n bowen python scripts/build_fold_targets.py
conda run -n bowen python scripts/extract_dinov3.py --device cuda
conda run -n bowen python scripts/run_smoke_cell.py --device cuda
conda run -n bowen python scripts/run_matrix.py --device cuda
conda run -n bowen python scripts/freeze_evaluation.py
conda run -n bowen python scripts/evaluate_mechanism.py
conda run -n bowen python scripts/evaluate_pcr.py
conda run -n bowen python scripts/generate_private_sha_manifest.py
conda run -n bowen python scripts/finalize.py
conda run -n bowen python scripts/audit_public_artifacts.py
```

PyRadiomics 3.1.0 runs in a hash-recorded Python 3.9 side environment because
the main `bowen` environment uses Python 3.11. The orchestration, data checks,
model training, and evaluation remain in `bowen`.

The V1 experiment and its `NO_GO` decision are immutable. V2 may stop at ROI,
target, optimization, or mechanism gates without opening a pCR-bearing file.
