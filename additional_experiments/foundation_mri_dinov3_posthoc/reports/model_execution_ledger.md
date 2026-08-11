# DINOv3 post-hoc execution ledger

This ledger is intentionally metric-free until the frozen reporting command
publishes aggregate results. Patient identifiers, predictions, selections,
cache paths, checkpoint paths, and progress logs are private and ignored.

## Lineage

- Parent publication commit: `98dfc5ab30804f288ff85f8d2b8c10905afbbe8b`
- Extension branch: `feature/foundation-mri-dinov3-posthoc`
- Original formal candidate set: unchanged
- Extension status: user-requested, post-publication, post-hoc sensitivity
- Pretraining-contamination status: unknown, not provably zero

## Model

- Official model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Hugging Face revision: `5931719e67bbdb9737e363e781fb0c67687896bc`
- Official GitHub revision: `6876159a11b4df116f30f667f8c9888617df0751`
- Checkpoint SHA-256: `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`
- Parameters: 85,660,416
- Representation: CLS plus mean non-register patch token, 1,536 dimensions
- License: DINOv3 custom license; checkpoint not redistributed

## Frozen data and evaluation

- Split seed: 2026
- Split manifest SHA-256: `143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38`
- C1B-H cache manifest SHA-256: `672ad7436b19f30a89640a2b36504f1e7fbaaff83fd07bc058c008b204d2a3c9`
- Clinical input SHA-256: `b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436`
- Formal FTV/radiomics input SHA-256: `26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d`
- Full cohort: 808 patients; complete-case sensitivity: 375 patients
- Timings: T0, T0-T1, T0-T2
- pCR cells: 36 DINOv3 identities
- Representation probes: 4 binary phenotype, 2 subtype, 14 FTV/delta-FTV identities
- Paired comparisons: 84 specifications, each with AUROC, AUPRC, and Brier

## Runtime

The following was recorded only after the corresponding metric-free completion
markers existed.

- Python: 3.11.14
- NumPy: 2.2.6; pandas: 2.3.3; scikit-learn: 1.8.0; SciPy: 1.16.3
- PyTorch: 2.9.1+cu130; CUDA runtime: 13.0; cuDNN: 9.13.0
- transformers: 5.2.0; safetensors: 0.7.0
- matplotlib: 3.10.8; joblib: 1.5.3; threadpoolctl: 3.6.0; pytest: 8.4.2
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- NVIDIA driver: 580.76.05
- Formal CPU selection environment: `PYTHONHASHSEED=2026`,
  `FOUNDATION_MRI_SELECTION_WORKERS=1`, `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Formal extraction environment: CUDA device 0, bfloat16 inference,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, Hugging Face/Transformers offline,
  implicit token and telemetry disabled

## Immutable lock and feature receipts

- Model-input lock: `ca10212a2fc4c540a261d2d4b4daca4f3a97e8931edb3a0fd0ebbedd6cb18b80`
  (mode 0444; frozen before the first patient shard)
- Evaluation lock: `7732828579d8159f927ce821348fcfe24d430b37d4218db9464c9bce881d1b38`
  (mode 0444; frozen before any DINOv3 outcome evaluation)
- Frozen feature asset: shape `[808,4,2,1536]`, float32, all finite, SHA-256
  `cffebbeefea0a9c08dcc41d3eb9e43bbb8e078449ddf16dc608893570bfd0c45`
- Extraction contract SHA-256:
  `5b268da89eae4d90b7271b25e0ae2a3e1a6dc3e500b8fc5cf64b82d6005230c0`
- Extraction execution SHA-256:
  `8792cf5f757008d2859818fec6168fa5921e78d70f1d5130c43de091c7045ebe`
- Extraction wall time: 1,426.14 seconds; 808/808 private shards completed;
  `outcome_fields_consumed=[]`

## Formal evaluation completion

- pCR baseline: `COMPLETED`, exit 0; 36 identities, 180 outer folds,
  3,240 validation candidates; maximum observed iteration count 50/20,000;
  no warning/error event
- Baseline receipt SHA-256:
  `88e1ba5e227d8e90a064678499aed0e55d1c2584711a2fabf29bf49c7c2cd10c`
- Baseline public aggregate SHA-256:
  `1faad23cea5071b102c158231d59adb33c31141087e8217924a9a178cf64aa78`
  (72 data rows)
- Phenotype/subtype/FTV probes: `COMPLETED`, exit 0; 20 identities,
  100 outer folds, 1,100 validation candidates and 720 explicit one-vs-rest
  estimators; maximum observed iterations binary/subtype/Ridge = 56/24/198;
  no warning/error event
- Probe receipt SHA-256:
  `ec36ac19e37e146fccbdaaafa95c6b5f62401cac4c645e0ec9f60611b3bfdd19`
- Probe public aggregate SHA-256 values: phenotype
  `b5b1c3cf69c4cc40b89ba767aafb7b17d8581c3f12262795eb544524000cff59`;
  subtype `1029949b6baabcd5b996ef584d554f153b95b79092188c2f9f21bed41abbbcec`;
  FTV `d8a1b3c9b42f2e16401557a909af2fd2990ab0cf24cb17549d308cd154516c12`
- All private predictions, selections, progress logs, receipts, and feature
  assets remain mode 0600 and are excluded by the experiment-local ignore rules.

## Reporting completion

- Formal reporter: `COMPLETED`, exit 0; 36 pooled pCR cells, 84 paired
  comparison specifications, 252 metric rows, 5,000/5,000 valid patient-level
  bootstrap replicates for every metric row
- Reporting provenance SHA-256:
  `e2f84967d1a050dca25273ab4154d9b085f4f03699625c611b973481c4c2f1b6`
- Public artifacts: paired comparison CSV
  `69b064f847c792c992982646bad341fa4e7e85c0df5b1cd309ae14dc1b7bad68`;
  results summary JSON
  `5d9818550cd7135ade515711e6aa54172cf840b00f9589b16a4e943dcc71b9b8`;
  scientific final report
  `44666a0d85e66d3f436173291fe086ef4ad684d9ee30403819f80a910e317750`;
  timing figure
  `824c71053757d9eed508d87b59a338d820e7050209000a6d60f155b60c1faff8`;
  comparison figure
  `429eb0ba756c86434724f3aa3be42fd1f7d20b870995b2b7f81ae233493d6e6d`
- The complete 12-question interpretation is in
  [`scientific_interpretation_addendum.md`](scientific_interpretation_addendum.md),
  with a separate public provenance marker. It is explicitly post-publication
  interpretation and does not alter the hash-bound scientific output bundle.

## Publication boundary

No raw MRI, patient-level source table, private prediction/selection/progress
file, frozen feature asset, or checkpoint is eligible for Git staging. The
substantive content commit contains only code, configs, aggregate metrics,
figures, reports, and public-safe provenance. Branch, content commit SHA and
the real push status are inserted into `final_report.md` only by the frozen
post-commit Git handoff finalizer; this avoids a self-referential report hash.
