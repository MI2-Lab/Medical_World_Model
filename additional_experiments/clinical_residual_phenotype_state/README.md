# Clinical-residual phenotype state pilot

This experiment implements Goal F on branch
`feature/clinical-residual-phenotype-state`. It keeps the confirmed C1B-H,
DCE7, fixed 64-mm LOCAL response pathway and splits a fixed 192-D MRI state
into a 96-D response state (`z_R`) and a 96-D single-query phenotype state
(`z_P`). FTV grounds only `z_R`. The F2 arm applies small linear HR/HER2
gradient-reversal heads only to `z_P`; treatment remains a valid transition
condition.

## Evidence basis

The implementation was designed after auditing the canonical LOCAL response
confirmation, MRI-clinical complementarity, compact fusion, classical DCE,
spatial heterogeneity, foundation MRI, DINOv3, and non-FTV phenotype
experiments. The combined evidence supports LOCAL as a burden/response state,
shows weak current MRI complementarity, finds some traditional-DCE non-FTV
signal, and favors learned localization over another fixed mean/std statistic.
It does not support broader BPE context or generic encoder replacement, so
neither is introduced here.

## Outcome firewall and immutable boundaries

- Representation training reads a private manifest with exactly
  `patient_id,label_hr,label_her2,label_mp,arm`; it never parses pCR.
- `PREREGISTRATION_LOCK.json` freezes the representation config and training,
  model, loss, data, and export code before formal training.
- All 20 F1/F2 exports and all 10 confirmed LOCAL3 controls must exist before
  `EVALUATION_LOCK.json` can be created.
- The evaluation lock hashes every feature/metadata asset, selected checkpoint
  and selection binding, export status, evaluation config, and evaluator/report
  source. The clinical-label loader refuses to open until that lock verifies.
- Patient-level OOF predictions and bootstrap draws remain under the ignored
  `predictions/` tree. Public outputs are aggregate tables and hashes only.

## Formal workflow

```bash
python scripts/build_training_profiles.py
python scripts/freeze_preregistration.py
python scripts/run_matrix.py --mode train --devices 0,1,2
python scripts/run_matrix.py --mode export --devices 0,1,2
python scripts/freeze_evaluation.py
python scripts/evaluate_frozen.py
python scripts/generate_report.py
```

The matched response evaluator exactly reproduces the confirmed LOCAL3
two-seed static and delta macro-Spearman values before comparing `z_R`.
Future-image retention compares T1→T2 and T2→T3 prediction against persistence
in the same EMA-target-projector coordinate system; T0→T1 is excluded because
the frozen export does not contain EMA-target T0 context.

## Primary outputs

- `metrics/state_diagnostics.csv`
- `metrics/state_dimension_diagnostics.csv`
- `metrics/state_covariance_eigenspectra.csv`
- `metrics/nearest_neighbor_stability.csv`
- `metrics/response_metrics.csv`
- `metrics/phenotype_probes.csv`
- `metrics/pcr_metrics.csv`
- `metrics/paired_bootstrap_effects.csv`
- `metrics/decision_summary.json`
- `reports/final_report.md` (Chinese; answers all 12 Goal-F questions)

The primary interpretation remains a two-seed, five-fold internal OOF pilot;
it is not external validation.
