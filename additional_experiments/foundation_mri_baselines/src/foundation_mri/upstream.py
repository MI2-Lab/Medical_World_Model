"""Hash-locked adapters to audited cohort/cache and spatial primitives."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]
STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
POOLING_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_spatial_pooling_bottleneck_audit"
    / "src"
)

EXPECTED_POOLING_SHA256 = (
    "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54"
)
EXPECTED_POOLING_INIT_SHA256 = (
    "55dbb7a79f6248075464cb983617296bc71cd391b0c08d956c077f3ce0c75584"
)
EXPECTED_FOLD_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "672ad7436b19f30a89640a2b36504f1e7fbaaff83fd07bc058c008b204d2a3c9"
)
EXPECTED_STAGE_B_CONTRACTS_SHA256 = (
    "48d7738b6764780ba2e784f826be44ac718fdbb0beb526ec31c3c5525cba4bf9"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _load_pooling() -> Any:
    package = POOLING_SRC / "c1b_spatial_audit"
    observed = {
        "__init__.py": file_sha256(package / "__init__.py"),
        "pooling.py": file_sha256(package / "pooling.py"),
    }
    expected = {
        "__init__.py": EXPECTED_POOLING_INIT_SHA256,
        "pooling.py": EXPECTED_POOLING_SHA256,
    }
    if observed != expected:
        raise ImportError(
            f"audited spatial pooling source drifted: expected={expected}, "
            f"observed={observed}"
        )
    _prepend(POOLING_SRC)
    module = importlib.import_module("c1b_spatial_audit.pooling")
    if Path(inspect.getfile(module.fixed_physical_local_weights)).resolve() != (
        package / "pooling.py"
    ).resolve():
        raise ImportError("spatial primitive resolved outside the audited source")
    return module


def _load_stage_data() -> Any:
    contracts_source = STAGE_B_SRC / "c1b_stage_b" / "contracts.py"
    observed_contracts = file_sha256(contracts_source)
    if observed_contracts != EXPECTED_STAGE_B_CONTRACTS_SHA256:
        raise ImportError(
            "audited Stage-B contracts source drifted: "
            f"expected={EXPECTED_STAGE_B_CONTRACTS_SHA256}, "
            f"observed={observed_contracts}"
        )
    _prepend(STAGE_B_SRC)
    module = importlib.import_module("c1b_stage_b.data")
    expected = STAGE_B_SRC / "c1b_stage_b" / "data.py"
    if Path(inspect.getfile(module.load_dce7)).resolve() != expected.resolve():
        raise ImportError("C1B loader resolved outside the Stage-B source")
    return module


POOLING = _load_pooling()
STAGE_DATA = _load_stage_data()

fixed_physical_local_weights = POOLING.fixed_physical_local_weights
global_average_pool = POOLING.global_average_pool
weighted_average_pool = POOLING.weighted_average_pool
load_dce7 = STAGE_DATA.load_dce7
read_cache_manifest = STAGE_DATA.read_cache_manifest
read_fold_manifest = STAGE_DATA.read_fold_manifest


def upstream_contract() -> dict[str, object]:
    return {
        "pooling_sha256": EXPECTED_POOLING_SHA256,
        "pooling_init_sha256": EXPECTED_POOLING_INIT_SHA256,
        "fold_manifest_sha256": EXPECTED_FOLD_SHA256,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "stage_b_contracts_sha256": EXPECTED_STAGE_B_CONTRACTS_SHA256,
        "stage_b_data_source_sha256": file_sha256(
            STAGE_B_SRC / "c1b_stage_b" / "data.py"
        ),
    }


__all__ = [
    "EXPECTED_CACHE_MANIFEST_SHA256",
    "EXPECTED_FOLD_SHA256",
    "EXPECTED_POOLING_SHA256",
    "EXPECTED_STAGE_B_CONTRACTS_SHA256",
    "EXPERIMENT_ROOT",
    "REPO_ROOT",
    "file_sha256",
    "fixed_physical_local_weights",
    "global_average_pool",
    "load_dce7",
    "read_cache_manifest",
    "read_fold_manifest",
    "upstream_contract",
    "weighted_average_pool",
]
