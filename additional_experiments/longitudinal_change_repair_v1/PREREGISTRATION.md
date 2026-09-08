# Frozen protocol

* Population: the existing 808 I-SPY2 five-fold manifest; only the 375
  measurement-overlap patients contribute FTV endpoints. I-SPY1 is optionally
  included in self-supervised training only and has no FTV supervision.
* Primary observed target: `log1p(FTV_end)-log1p(FTV_start)`, fitted and
  standardized from outer-train patients only.
* R0 has no FTV loss; R1 adds static FTV loss; R2 adds observed-change loss.
  All have identical architecture and initialization within seed/fold.
* Stage F has F1/F2 initialized from R1 and F3/F4 from R2. F1/F3 duplicate the
  current source; F2/F4 use the real preceding source. It predicts T1->T2 and
  T2->T3 FTV changes without future MRI as input.
* Selection uses validation change MAE only. Outer-test metrics are exploratory
  developmental replication, not an independent confirmation.
* The Stage-R gate requires an observed-change validation R2 > 0 and a positive
  real-history minus current-only change in both seeds. Stage F cannot start
  otherwise.
* Primary Stage-F comparison is F2-F1 and F4-F3 per horizon. It reports paired
  patient bootstrap confidence intervals (10,000 draws) and Holm adjustment for
  the two horizons. Counterfactuals never select checkpoints.
