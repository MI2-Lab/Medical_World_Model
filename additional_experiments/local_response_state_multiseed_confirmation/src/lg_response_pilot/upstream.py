"""Hash-locked imports of the sealed G3 model/objective and pooling audit.

No encoder, projector, transition, EMA, objective, or fractional-overlap
implementation is copied into this pilot.  Imports fail closed unless they
resolve to the exact completed source files and those files retain their
frozen SHA-256 digests.
"""

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
G3_ROOT = REPO_ROOT / "additional_experiments" / "g3_multiseed_generalization"
G3_SRC = G3_ROOT / "src"
POOLING_AUDIT_ROOT = (
    REPO_ROOT / "additional_experiments" / "c1b_spatial_pooling_bottleneck_audit"
)
POOLING_AUDIT_SRC = POOLING_AUDIT_ROOT / "src"

G3_SOURCE_SHA256 = {
    "__init__.py": "c18fa03739e604a77018975ec1d2e7ed00339d8b6a529562446c845e9200b9b8",
    "config.py": "4460ce3413e2cb936a6fd3cbb7f16224af3af286b6784933688cd12d0ec47516",
    "data.py": "15b4b68ad45c935e313b893b0ce849877311c98d6c5c0c45495e8e9200240943",
    "model.py": "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9",
    "targets.py": "28fbf66f93c8541dfa5ecc7ebcf65d4143a9a605b3ce98be48355d5ab679ffac",
    "training.py": "76f9108df0ca8c0ff69e514cff3bab1d5e316d946da60c5f530dd7b9706d3815",
}
AUDITED_POOLING_SHA256 = (
    "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54"
)
AUDITED_SOURCE_SHA256 = {
    "__init__.py": "55dbb7a79f6248075464cb983617296bc71cd391b0c08d956c077f3ce0c75584",
    "pooling.py": AUDITED_POOLING_SHA256,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_sources() -> dict[str, Any]:
    observed_g3: dict[str, str] = {}
    for name, expected in G3_SOURCE_SHA256.items():
        path = G3_SRC / "dgrs" / name
        if not path.is_file():
            raise ImportError(f"sealed G3 source is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ImportError(
                f"sealed G3 source hash drifted for {name}: "
                f"expected {expected}, observed {actual}"
            )
        observed_g3[name] = actual
    observed_audit: dict[str, str] = {}
    audit_package = POOLING_AUDIT_SRC / "c1b_spatial_audit"
    for name, expected in AUDITED_SOURCE_SHA256.items():
        path = audit_package / name
        if not path.is_file():
            raise ImportError(f"audited pooling source is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ImportError(
                f"audited pooling source hash drifted for {name}: "
                f"expected {expected}, observed {actual}"
            )
        observed_audit[name] = actual
    return {"g3": observed_g3, "audited_pooling": observed_audit}


SOURCE_VERIFICATION = verify_upstream_sources()
for source in (G3_SRC, POOLING_AUDIT_SRC):
    value = str(source.resolve())
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

_MODEL = importlib.import_module("dgrs.model")
_TRAINING = importlib.import_module("dgrs.training")
_POOLING = importlib.import_module("c1b_spatial_audit.pooling")
_G3_PACKAGE = importlib.import_module("dgrs")
_AUDIT_PACKAGE = importlib.import_module("c1b_spatial_audit")

_EXPECTED_MODEL = (G3_SRC / "dgrs" / "model.py").resolve()
_EXPECTED_TRAINING = (G3_SRC / "dgrs" / "training.py").resolve()
_EXPECTED_POOLING = (
    POOLING_AUDIT_SRC / "c1b_spatial_audit" / "pooling.py"
).resolve()
if Path(str(getattr(_G3_PACKAGE, "__file__", ""))).resolve() != (
    G3_SRC / "dgrs" / "__init__.py"
).resolve():
    raise ImportError("dgrs package did not resolve to the sealed G3 source")
if Path(str(getattr(_AUDIT_PACKAGE, "__file__", ""))).resolve() != (
    POOLING_AUDIT_SRC / "c1b_spatial_audit" / "__init__.py"
).resolve():
    raise ImportError("c1b_spatial_audit package did not resolve to audited source")
if Path(inspect.getfile(_MODEL.DGRSWorldModel)).resolve() != _EXPECTED_MODEL:
    raise ImportError("DGRSWorldModel did not resolve to the sealed G3 source")
if Path(inspect.getfile(_TRAINING.DGRSObjective)).resolve() != _EXPECTED_TRAINING:
    raise ImportError("DGRSObjective did not resolve to the sealed G3 source")
if Path(inspect.getfile(_POOLING.fixed_physical_local_weights)).resolve() != _EXPECTED_POOLING:
    raise ImportError("fixed_physical_local_weights did not resolve to the audited source")
if Path(inspect.getfile(_POOLING.weighted_average_pool)).resolve() != _EXPECTED_POOLING:
    raise ImportError("weighted_average_pool did not resolve to the audited source")

DGRSOutput = _MODEL.DGRSOutput
DGRSWorldModel = _MODEL.DGRSWorldModel
DGRSObjective = _TRAINING.DGRSObjective
audited_expected_feature_shape = _POOLING.expected_feature_shape
fixed_physical_local_weights = _POOLING.fixed_physical_local_weights
weighted_average_pool = _POOLING.weighted_average_pool


__all__ = [
    "AUDITED_POOLING_SHA256",
    "AUDITED_SOURCE_SHA256",
    "DGRSObjective",
    "DGRSOutput",
    "DGRSWorldModel",
    "G3_SOURCE_SHA256",
    "SOURCE_VERIFICATION",
    "audited_expected_feature_shape",
    "file_sha256",
    "fixed_physical_local_weights",
    "verify_upstream_sources",
    "weighted_average_pool",
]
