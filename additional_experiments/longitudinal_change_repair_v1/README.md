# Longitudinal Change Repair v1

This is an outcome-free, five-fold **developmental replication** that separates
observed-change decoding from future-change forecasting.  It consumes only the
locked DCE8 cache and the established I-SPY2 fold manifest.  pCR is not loaded
by any training module.

Run `scripts/preflight.py` before training.  `scripts/run_matrix.py --stage r`
trains the 30 Stage-R cells; it writes `stage_r_gate.json`.  Stage F is refused
until that gate passes.  The implementation never overwrites a prior run.

Patient-level outputs are written under `private_results/` (gitignored). Public
reporting uses only aggregate CSVs and the generated Chinese report.
