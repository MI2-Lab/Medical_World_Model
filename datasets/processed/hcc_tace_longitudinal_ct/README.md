# HCC-TACE Longitudinal CT Preprocessing

This folder contains manifests generated for longitudinal CT latent prediction.

## Files

- `study_manifest.csv`: one selected CT series per study, plus SEG path when available
- `longitudinal_pairs_adjacent.csv`: adjacent study pairs per patient
- `longitudinal_pairs_first_last.csv`: first-to-last study pair per patient
- `summary.json`: preprocessing summary and counts

## Selection Logic

- Keeps CT series only for latent prediction inputs/targets
- Selects one best CT series per study using DICOM metadata and a heuristic score
- Sorts studies by `StudyDate`
- Attaches SEG series from the same study when present
- Splits data deterministically by patient into train/val/test
