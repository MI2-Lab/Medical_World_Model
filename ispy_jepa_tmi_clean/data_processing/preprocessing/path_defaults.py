from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
LOCAL_METADATA_ROOT = PROJECT_ROOT / "metadata"


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser()


def ispy2_raw_root() -> Path:
    return env_path("ISPY2_RAW_ROOT", LOCAL_DATA_ROOT / "I-SPY2")


def ispy1_raw_root() -> Path:
    return env_path("ISPY1_RAW_ROOT", LOCAL_DATA_ROOT / "I-SPY1")


def ispy2_preprocessed_root() -> Path:
    return env_path("ISPY2_PREPROCESSED_ROOT", LOCAL_DATA_ROOT / "Preprocessed" / "I-SPY2")


def ispy1_preprocessed_root() -> Path:
    return env_path("ISPY1_PREPROCESSED_ROOT", LOCAL_DATA_ROOT / "Preprocessed" / "I-SPY1")


def dcm2niix_path() -> Path:
    found = shutil.which("dcm2niix")
    return env_path("DCM2NIIX", found or "dcm2niix")


def breastdcedl_root() -> Path:
    return env_path("BREASTDCEDL_ROOT", LOCAL_METADATA_ROOT)


def breastdcedl_metadata_csv() -> Path:
    return env_path(
        "BREASTDCEDL_METADATA_CSV",
        breastdcedl_root() / "BreastDCEDL_metadata_min_crop.csv",
    )
