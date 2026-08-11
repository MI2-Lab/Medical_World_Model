"""Repository-local paths shared by the isolated extension."""

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BASE_EXPERIMENT_ROOT = REPOSITORY_ROOT / "additional_experiments/foundation_mri_baselines"
BASE_SOURCE_ROOT = BASE_EXPERIMENT_ROOT / "src"


__all__ = [
    "BASE_EXPERIMENT_ROOT",
    "BASE_SOURCE_ROOT",
    "EXPERIMENT_ROOT",
    "REPOSITORY_ROOT",
]

