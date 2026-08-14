"""Hash-locked access to the frozen encoder, LOCAL pooling, SIGReg, and transition.

The pilot adds a factorized state but does not copy or mutate the completed
C1B-H encoder and pooling implementations.  Every imported source is verified
before Python is allowed to import it.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

from .contracts import file_sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]
G3_SRC = REPO_ROOT / "additional_experiments" / "g3_multiseed_generalization" / "src"
POOLING_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_spatial_pooling_bottleneck_audit"
    / "src"
)
TRANSITION_PATH = REPO_ROOT / "ispy_jepa_tmi_clean" / "corejepa" / "models" / "transition.py"

G3_HASHES: Mapping[str, str] = {
    "dgrs/__init__.py": "c18fa03739e604a77018975ec1d2e7ed00339d8b6a529562446c845e9200b9b8",
    "dgrs/model.py": "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9",
    "dgrs/training.py": "76f9108df0ca8c0ff69e514cff3bab1d5e316d946da60c5f530dd7b9706d3815",
}
POOLING_HASHES: Mapping[str, str] = {
    "c1b_spatial_audit/__init__.py": "55dbb7a79f6248075464cb983617296bc71cd391b0c08d956c077f3ce0c75584",
    "c1b_spatial_audit/pooling.py": "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54",
}
TRANSITION_SHA256 = "a4d7741460938f41e090255045717b3b072e9d555e15eaa6357aa3a6839b531e"


def verify_upstream_sources() -> dict[str, Any]:
    observed: dict[str, Any] = {"g3": {}, "pooling": {}}
    for relative, expected in G3_HASHES.items():
        source = G3_SRC / relative
        actual = file_sha256(source)
        if actual != expected:
            raise ImportError(f"frozen G3 source drifted: {relative}")
        observed["g3"][relative] = actual
    for relative, expected in POOLING_HASHES.items():
        source = POOLING_SRC / relative
        actual = file_sha256(source)
        if actual != expected:
            raise ImportError(f"audited LOCAL pooling source drifted: {relative}")
        observed["pooling"][relative] = actual
    transition = file_sha256(TRANSITION_PATH)
    if transition != TRANSITION_SHA256:
        raise ImportError("frozen clinical/treatment transition source drifted")
    observed["conditioned_transition"] = transition
    return observed


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _load_transition() -> ModuleType:
    name = "_crps_frozen_conditioned_transition"
    spec = importlib.util.spec_from_file_location(name, TRANSITION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct frozen transition module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE_VERIFICATION = verify_upstream_sources()
_prepend(G3_SRC)
_prepend(POOLING_SRC)
_MODEL = importlib.import_module("dgrs.model")
_TRAINING = importlib.import_module("dgrs.training")
_POOLING = importlib.import_module("c1b_spatial_audit.pooling")
_TRANSITION = _load_transition()

for module, expected in (
    (_MODEL, (G3_SRC / "dgrs" / "model.py").resolve()),
    (_TRAINING, (G3_SRC / "dgrs" / "training.py").resolve()),
    (_POOLING, (POOLING_SRC / "c1b_spatial_audit" / "pooling.py").resolve()),
    (_TRANSITION, TRANSITION_PATH.resolve()),
):
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected:
        raise ImportError(f"upstream module resolved outside frozen source: {module}")

SpatialVisitEncoder3D = _MODEL.SpatialVisitEncoder3D
VisitProjector = _MODEL.VisitProjector
ImageOnlyCausalTransition = _MODEL.ImageOnlyCausalTransition
ImageTransition = _TRANSITION.ImageTransition
SIGReg = _TRAINING.SIGReg
patient_mean_ftv_loss = _TRAINING.patient_mean_ftv_loss
fixed_physical_local_weights = _POOLING.fixed_physical_local_weights
weighted_average_pool = _POOLING.weighted_average_pool
expected_feature_shape = _POOLING.expected_feature_shape

for value, expected_module in (
    (SpatialVisitEncoder3D, _MODEL),
    (VisitProjector, _MODEL),
    (ImageOnlyCausalTransition, _MODEL),
    (SIGReg, _TRAINING),
    (patient_mean_ftv_loss, _TRAINING),
    (fixed_physical_local_weights, _POOLING),
    (weighted_average_pool, _POOLING),
    (ImageTransition, _TRANSITION),
):
    if Path(inspect.getfile(inspect.unwrap(value))).resolve() != Path(expected_module.__file__).resolve():
        raise ImportError(f"upstream object resolved outside frozen module: {value}")


__all__ = [
    "G3_HASHES",
    "ImageOnlyCausalTransition",
    "ImageTransition",
    "POOLING_HASHES",
    "SIGReg",
    "SOURCE_VERIFICATION",
    "SpatialVisitEncoder3D",
    "TRANSITION_SHA256",
    "VisitProjector",
    "expected_feature_shape",
    "fixed_physical_local_weights",
    "patient_mean_ftv_loss",
    "verify_upstream_sources",
    "weighted_average_pool",
]
