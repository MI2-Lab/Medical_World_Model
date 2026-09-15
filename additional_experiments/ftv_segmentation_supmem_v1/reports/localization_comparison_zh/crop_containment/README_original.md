# Crop-placement containment audit

This audit asks whether a placed 32×96×96 `(z,y,x)` crop contains the released FTV lesion; it does **not** measure segmentation Dice. The arms are: `oracle` (released-mask centroid at each visit), `pipeline` (released T0 bbox centre projected to each visit), `learned_pervisit` (held-out predicted-mask centroid), `learned_t0anchor` (held-out predicted T0 centroid projected to each visit), and `auto_stub` (automatic enhancement-ROI centroid). Projection uses the upstream `_project_center`; containment uses upstream `crop_or_pad_cxyz` with the same formula as `scripts/data.py::volume_metrics`.

Arms 1–4 cover all 808 patients and use each prediction row's asserted unique held-out test fold. The raw-DCE automatic stub is evaluated on a fixed random 40-patient cohort (seed 2026); this bounded size matches the pre-existing stub baseline and is reported separately in every table. The six visual-comparison patients were evaluated additionally only to draw their boxes and do not enter auto-stub cohort summaries unless sampled. Empty truth masks have undefined containment: they remain in `crop_containment.json` with `empty_truth=true` but are excluded from the full non-empty-truth table. The all-four-visit table further requires non-empty truth at every visit. This is an easier subset, so it should not be compared as though it were full-cohort performance.

`crop_grid.png` reuses the six visual-comparison cases. Each cell has axial and coronal MIPs: coronal z is essential because z margins are often tight. Physical crop FOV is annotated in every cell: although the voxel count is fixed, spacing means its physical dimensions vary by patient. Magenta lesion pixels are outside the learned per-visit crop.

## Full non-empty-truth cohort

| Visit | Arm | n | Median | IQR | <0.9 | <0.5 |
|---|---|---:|---:|---:|---:|---:|
| T0 | oracle | 808 | 0.973 | 0.865–1.000 | 30.8% | 2.6% |
| T0 | pipeline | 808 | 0.977 | 0.827–1.000 | 33.9% | 5.1% |
| T0 | learned_pervisit | 808 | 0.928 | 0.738–0.997 | 44.3% | 14.2% |
| T0 | learned_t0anchor | 808 | 0.928 | 0.738–0.997 | 44.3% | 14.2% |
| T0 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |
| T1 | oracle | 806 | 0.979 | 0.877–1.000 | 29.7% | 3.3% |
| T1 | pipeline | 806 | 0.879 | 0.598–0.995 | 52.5% | 18.7% |
| T1 | learned_pervisit | 806 | 0.929 | 0.725–0.999 | 45.3% | 16.6% |
| T1 | learned_t0anchor | 806 | 0.830 | 0.507–0.979 | 59.9% | 24.6% |
| T1 | auto_stub | 40 | 0.000 | 0.000–0.007 | 97.5% | 95.0% |
| T2 | oracle | 791 | 0.966 | 0.807–1.000 | 36.8% | 6.1% |
| T2 | pipeline | 791 | 0.807 | 0.483–0.988 | 59.3% | 25.8% |
| T2 | learned_pervisit | 791 | 0.750 | 0.268–0.979 | 62.2% | 34.1% |
| T2 | learned_t0anchor | 791 | 0.742 | 0.365–0.971 | 64.1% | 32.9% |
| T2 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |
| T3 | oracle | 761 | 0.956 | 0.805–1.000 | 40.6% | 6.8% |
| T3 | pipeline | 761 | 0.750 | 0.388–0.979 | 64.8% | 31.4% |
| T3 | learned_pervisit | 761 | 0.638 | 0.153–0.940 | 70.8% | 42.3% |
| T3 | learned_t0anchor | 761 | 0.665 | 0.292–0.950 | 71.5% | 38.0% |
| T3 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |

## All-four-visits-non-empty subset

| Visit | Arm | n | Median | IQR | <0.9 | <0.5 |
|---|---|---:|---:|---:|---:|---:|
| T0 | oracle | 753 | 0.967 | 0.851–1.000 | 32.8% | 2.8% |
| T0 | pipeline | 753 | 0.966 | 0.817–1.000 | 36.0% | 5.4% |
| T0 | learned_pervisit | 753 | 0.917 | 0.722–0.996 | 46.3% | 14.7% |
| T0 | learned_t0anchor | 753 | 0.917 | 0.722–0.996 | 46.3% | 14.7% |
| T0 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |
| T1 | oracle | 753 | 0.975 | 0.862–1.000 | 31.7% | 3.6% |
| T1 | pipeline | 753 | 0.862 | 0.586–0.992 | 55.0% | 19.3% |
| T1 | learned_pervisit | 753 | 0.922 | 0.723–0.997 | 46.9% | 16.7% |
| T1 | learned_t0anchor | 753 | 0.822 | 0.498–0.971 | 61.9% | 25.4% |
| T1 | auto_stub | 40 | 0.000 | 0.000–0.007 | 97.5% | 95.0% |
| T2 | oracle | 753 | 0.961 | 0.803–1.000 | 38.1% | 6.4% |
| T2 | pipeline | 753 | 0.790 | 0.467–0.982 | 60.8% | 26.4% |
| T2 | learned_pervisit | 753 | 0.755 | 0.294–0.978 | 62.4% | 33.9% |
| T2 | learned_t0anchor | 753 | 0.732 | 0.347–0.961 | 65.3% | 33.5% |
| T2 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |
| T3 | oracle | 753 | 0.955 | 0.802–1.000 | 40.9% | 6.9% |
| T3 | pipeline | 753 | 0.744 | 0.388–0.976 | 65.1% | 31.5% |
| T3 | learned_pervisit | 753 | 0.639 | 0.174–0.940 | 70.8% | 42.1% |
| T3 | learned_t0anchor | 753 | 0.661 | 0.292–0.947 | 71.8% | 38.1% |
| T3 | auto_stub | 40 | 0.000 | 0.000–0.000 | 100.0% | 100.0% |

## Can the released mask be dropped for crop placement?

- **T0: No.** Pipeline <0.9 failure is 33.9%; learned per-visit is 44.3%; auto stub is 100.0% (fixed n=40).
- **T1: No.** Pipeline <0.9 failure is 52.5%; learned per-visit is 45.3%; auto stub is 97.5% (fixed n=40).
- **T2: No.** Pipeline <0.9 failure is 59.3%; learned per-visit is 62.2%; auto stub is 100.0% (fixed n=40).
- **T3: No.** Pipeline <0.9 failure is 64.8%; learned per-visit is 70.8%; auto stub is 100.0% (fixed n=40).

The distribution figure (`containment_distribution.png`) shows the full non-empty-truth ECDFs, including 0.5 and 0.9 thresholds. `escape_axis.png` assigns each <0.9 containment failure to x, y, z, or a combination using the stored per-axis crop margins. The machine-readable file contains centres, containment, signed per-side physical margins, escape axis, fold, and empty-truth status for every evaluated patient×visit×arm.
