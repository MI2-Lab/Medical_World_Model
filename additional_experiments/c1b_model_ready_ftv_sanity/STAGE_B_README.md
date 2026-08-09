# Stage B execution boundary

Stage B is implemented but is not launched by this repository change. Every
entry point first validates `STAGE_A_GO.json`; a missing sentinel, a `NO-GO`,
any failed sub-gate, or relaxed thresholds stops before a cache or output is
opened.

The formal data contract is a SHA-256-pinned JSON based on
`configs/stage_b_data_contract.example.json`. Build the two identifier-bearing
cache inventories only after Stage A GO with
`scripts/build_stage_b_cache_manifests.py`; both outputs must end in
`.private.csv`. Manifest creation accepts only the Stage-A-selected full C1B
table pinned by the GO sentinel, hashes every cache exactly once, and records
SHA-256, byte size, and nanosecond mtime. The cohort is fail-closed at exactly
808 primary plus 140 eligible I-SPY1 patients (948 unique patients).

Training is one `(arm, seed, fold)` per `scripts/train_stage_b.py`. The exact
matrix launcher is `scripts/run_stage_b_matrix.py`; it is read-only unless
`--execute` is supplied. It runs `L1,N1` before their paired `L3,N3` baselines
for seeds 2026/3026 and folds 0–4. The default is physical 4 / accumulation 8.
An OOM requires a new empty output root and a global all-arm 2/16 restart with
`--global-fallback-restart`.

After training, use `scripts/export_stage_b_features.py` (online
pre-projector `r` only), `scripts/run_stage_b_probes.py` (static natural FTV
and literal natural `FTV[t+1]-FTV[t]`), and `scripts/aggregate_stage_b.py`.
Aggregation refuses an incomplete 40-run matrix and produces paired effects,
DiD, optimization safety, and Stage B figures 7–14.

## Formal command template

The frozen legacy cache root is:

```text
/data/data/Preprocessed/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96
```

After the Stage A sentinel and the chosen full-scope C1B cache table exist,
set `C1B_TABLE` to the strategy named by the sentinel (`h` or `r`) and build
the pinned data contract:

```bash
REPO="$(git rev-parse --show-toplevel)"
EXP=$REPO/additional_experiments/c1b_model_ready_ftv_sanity
PY=${C1B_PYTHON:-python}
FOLD=/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv
LEGACY=/data/data/Preprocessed/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96
ELIG=$EXP/manifests/ispy1_base_eligibility_patients.private.csv
FTV=$REPO/additional_experiments/radiomics_next_change/data_audit/radiomics_transition_targets_raw.csv
OBS=$EXP/manifests/grounding_observability_manifest.private.csv
C1B_TABLE=$EXP/metrics/model_input_pipeline_h_all.private.csv
DATA_CONTRACT=$EXP/manifests/stage_b_data_contract.private.json

$PY $EXP/scripts/build_stage_b_cache_manifests.py \
  --stage-a-sentinel $EXP/STAGE_A_GO.json \
  --fold-manifest $FOLD \
  --fold-manifest-sha256 $(sha256sum $FOLD | cut -d' ' -f1) \
  --ispy1-eligibility-manifest $ELIG \
  --ispy1-eligibility-manifest-sha256 $(sha256sum $ELIG | cut -d' ' -f1) \
  --legacy-cache-root $LEGACY \
  --c1b-stage-a-cache-table $C1B_TABLE \
  --c1b-stage-a-cache-table-sha256 $(sha256sum $C1B_TABLE | cut -d' ' -f1) \
  --legacy-output $EXP/manifests/stage_b_legacy_cache.private.csv \
  --c1b-output $EXP/manifests/stage_b_c1b_cache.private.csv \
  --ftv-transition-table $FTV \
  --ftv-transition-table-sha256 $(sha256sum $FTV | cut -d' ' -f1) \
  --observability-manifest $OBS \
  --observability-manifest-sha256 $(sha256sum $OBS | cut -d' ' -f1) \
  --data-contract-output $DATA_CONTRACT \
  --summary-output $EXP/metrics/stage_b_data_contract_summary.json
```

Read-only matrix preflight (no training):

```bash
DATA_SHA=$(sha256sum $DATA_CONTRACT | cut -d' ' -f1)
$PY $EXP/scripts/run_stage_b_matrix.py \
  --stage-a-sentinel $EXP/STAGE_A_GO.json \
  --data-contract $DATA_CONTRACT \
  --data-contract-sha256 $DATA_SHA \
  --output-root $EXP/checkpoints/stage_b_bs4_accum8 \
  --devices cuda:0,cuda:1,cuda:2 \
  --physical-batch-size 4 --accumulation-steps 8 --workers 2
```

This formal preflight recomputes every pinned cache SHA-256 once and validates
the stat and archive envelope before any launch. The forty child processes do
not repeat full-file or schema-content hashing. Each data-worker read opens the
cache once, checks pinned size/mtime with `fstat` on that same descriptor,
checks the exact raw NPZ central-directory names, and materializes only the
model image plus the tiny schema-version and embedded-patient-ID members. A
touch, replacement, row/cache identity swap, renamed C1B file, duplicate/raw
alias member, or schema drift fails closed.

This assigns the ten `(seed,fold)` groups round-robin across three GPUs; each
GPU runs one group at a time and each group runs `L1,N1,L3,N3` sequentially.
The backward-compatible single-device form is `--device cuda`.

Formal 40-run launch is the same command with `--execute`. If and only if the
4/8 matrix stops on OOM, preserve that failed root and start all arms in a new
root using `--physical-batch-size 2 --accumulation-steps 16
--global-fallback-restart --execute`.

C1B cache schema 3 remains fully validated by the unchanged Stage A validator.
The selective Stage B adapter requires integer scalar schema version 3, binds
manifest ID = embedded Unicode ID = SHA-256-tokenized filename, preflights NPY
headers and member byte sizes before allocation, accepts only finite clipped
`[4,7,112,176,160]` float32 C-order `image`, and returns only that array.
Future-support, valid-mask, affine, normalization, and provenance sidecars are
listed for exact-envelope checking but are never opened or materialized by a
training worker.
