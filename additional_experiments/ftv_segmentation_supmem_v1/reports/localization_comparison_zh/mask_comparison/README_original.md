# I-SPY2 lesion-localization visual comparison

Six cases were selected mechanically from the 808-patient completed frozen-memory evaluation. Quality is each patient's mean held-out supervised Dice over T0/T1/T2 (T3 is excluded because it is degenerate). Within pCR = 0 and pCR = 1 separately, candidates were required to have non-empty released truth on all four visits; the nearest cases to the 90th, 50th, and 10th percentile values were selected, with patient ID as a stable tie-break. pCR was used only for this selection and row labels. `selection_context.png` shows the full distribution (including excluded empty-truth cases).

Each supervised prediction was re-run from `runs/frozen_memory/fold_k/best.pt`, where `k` is that patient's unique `split=test` fold in the supplied CV CSV; this is asserted by the generator. The main grid displays the full native-FOV axial slice having the largest released-mask area at that visit. The grayscale image is the cached early−pre channel, bilinearly upsampled from 256×256 to native in-plane resolution. In-panel Dice values are slice Dice; the table below is volume Dice. An absent dashed contour is explicitly annotated with automatic-ROI centroid displacement.

## Selected-case volume Dice

| Patient | pCR | held-out fold | quality (T0–T2) | quantile | T0 pred / auto | T1 pred / auto | T2 pred / auto | T3 pred / auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACRIN-6698-654064 | 0 | 3 | 0.744 | 90th | 0.784 / 0.000 | 0.745 / 0.000 | 0.704 / 0.000 | 0.784 / 0.000 |
| ACRIN-6698-227055 | 0 | 3 | 0.548 | 50th | 0.676 / 0.000 | 0.717 / 0.001 | 0.253 / 0.019 | 0.487 / 0.007 |
| ACRIN-6698-266840 | 0 | 4 | 0.255 | 10th | 0.555 / 0.000 | 0.032 / 0.000 | 0.178 / 0.000 | 0.001 / 0.000 |
| ACRIN-6698-584561 | 1 | 0 | 0.621 | 90th | 0.816 / 0.000 | 0.592 / 0.000 | 0.454 / 0.000 | 0.421 / 0.000 |
| ISPY2-252748 | 1 | 4 | 0.431 | 50th | 0.557 / 0.000 | 0.600 / 0.000 | 0.137 / 0.000 | 0.071 / 0.000 |
| ACRIN-6698-237158 | 1 | 3 | 0.177 | 10th | 0.121 / 0.000 | 0.407 / 0.011 | 0.002 / 0.000 | 0.332 / 0.000 |

## Cohort context

The completed 808-patient frozen-memory mean Dice is **0.607/0.526/0.309/0.216** at T0/T1/T2/T3, respectively (from `metrics/frozen_memory_summary.json`). The separately evaluated 40-patient automatic-ROI stub baseline has median Dice 0.0000 at every visit and median centroid displacement 97 mm (from `../auto_roi_localization_eval/metrics/summary.json`).

The automatic-ROI metrics in `../../metrics/visual_comparison_cases.json` were recomputed on these cases with `scripts/data.py::volume_metrics`, exactly matching the supervised metric definition.

Eligibility counts: pCR = 0: 516; pCR = 1: 237; total: 753.
