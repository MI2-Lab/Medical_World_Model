# Bilateral MRI Context pCR Ceiling v1

This experiment tests whether the original bilateral DCE-MRI acquisition contains
conditional pCR information that is absent from the lesion-centred LOCAL input.
It is an internal hypothesis-development experiment; it is not an independent
confirmatory analysis.

The representation extractor is outcome-blind and reads only raw bilateral DCE
images. The pCR evaluator is locked behind `INPUT_LOCK.json` and uses the frozen
clinical+FTV timing-safe fusion contract.

## Environment

```bash
conda run -n bowen python scripts/preflight_input.py
conda run -n bowen python scripts/extract_bilateral_dino.py --device cuda:0 --shard-index 0 --shards 3
```

The extraction script is resumable and writes only private tokenized caches.
