# DINOv3 MRI adapter + direct radiomics grounding

This experiment asks whether a trainable MRI-domain adapter over frozen DINOv3
slice summaries can produce pCR-relevant image information beyond clinical data
and FTV. Representation learning is outcome-blind. The pCR evaluator remains
fail-closed until all 75 D1/D2/D3 cells and all `[808,4,192]` state files have
been frozen into `EVALUATION_LOCK.json`.

Main commands use the Anaconda `bowen` environment:

```bash
conda run -n bowen python scripts/preflight.py
conda run -n bowen python scripts/extract_dinov3.py --device cuda
conda run -n bowen python scripts/build_radiomics_rois.py
conda run -n bowen python scripts/extract_radiomics.py --radiomics-python <py39-python>
conda run -n bowen python scripts/build_fold_targets.py
conda run -n bowen python scripts/run_matrix.py --device cuda
conda run -n bowen python scripts/freeze_evaluation.py
conda run -n bowen python scripts/evaluate_pcr.py
conda run -n bowen python scripts/finalize.py
```

PyRadiomics 3.1.0 runs in a hash-recorded Python 3.9 side environment because
the main `bowen` environment uses Python 3.11. The orchestration, data checks,
model training, and evaluation remain in `bowen`.
