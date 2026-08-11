from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
for source in (
    EXPERIMENT_ROOT / "src",
    REPOSITORY_ROOT / "additional_experiments/foundation_mri_baselines/src",
):
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)

