# DINOv3 MRI adapter + direct radiomics grounding V3

V3 tests whether FTV/volume-residualized DCE radiomics can be transferred into
a 192-D DINOv3 MRI image state without an FTV grounding loss. It reuses the
hash-locked V2 DINO summaries and fold-specific PCA16 targets; it does not rerun
PyRadiomics or DINO extraction.

The execution order is fail-closed: inheritance/preflight; one-seed five-fold
R025/R050/R100 pilot; matched state-probe pilot gate; fresh-seed 50-cell C0/RAD
matrix; outcome-blind mechanism lock; and conditional pCR evaluation only
after both locks.

All orchestration runs in Anaconda `bowen`. The model forward accepts only
`[B,4,7,32,2304]` frozen image summaries. Candidate FTV loss is exactly zero.

V2 and all V2 decisions remain immutable. Failure at any outcome-blind gate
leaves pCR unread.

## Frozen result

The preregistered pilot terminated as `DIRECT_RAD_WEIGHT_SCREEN_NO_GO`.
R025/R050/R100 each failed paired JEPA checkpoint safety in folds 0 and 1
after all eight epochs. Because every candidate had already become unable to
pass the required five-fold safety gate, the remaining nine cells were skipped
and neither the 50-cell matrix nor pCR evaluation was started.

The fixed V2 C0 states nevertheless reached 0.2861 macro Spearman with the
matched linear residual-radiomics probe. Thus V3 identifies an optimization
conflict in the registered direct-grounding update, not target infeasibility.
See `reports/final_report.md` and `pilot_gate.json` for the public result.
