"""Hash-locked Stage-B data, batching, and FTV-transform adapters."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

from .contracts import file_sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[3]
STAGEB_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
STAGEB_SRC = STAGEB_ROOT / "src"

STAGEB_HASHES: Mapping[str, str] = {
    "__init__.py": "2976c6d040c506b4f1b1db5374718d1c3edf341805d0e6a9d176f0c02fa37a47",
    "contracts.py": "48d7738b6764780ba2e784f826be44ac718fdbb0beb526ec31c3c5525cba4bf9",
    "data.py": "948a25aa00eeaf68a11f5a0bcf7c4d0c7592786a36ebb9bce472361745eebb59",
    "gate.py": "babb748a71eba0c36d802a8e15c861387d506de67cf754ec58ae96f1d3341555",
    "inputs.py": "40965e509afa059ce2674c7a7fde18cd9097e1eb00d05021738d8cf9f6346177",
    "targets.py": "06434db46cf76e6f39ff6eb1c476933885e90ed0a4c952dcc0a3477a25996c7b",
    "training.py": "2edf546628e447bdd1b9715f60f105d1a5952763bd782aabddbae298fae62f52",
    "upstream.py": "dfc03ab80590d1b57240a8ce210c75245bce4dd3bad9a4d655d8d63a1f96d54f",
}


def verify_stageb_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    package = STAGEB_SRC / "c1b_stage_b"
    for name, expected in STAGEB_HASHES.items():
        actual = file_sha256(package / name)
        if actual != expected:
            raise ImportError(f"sealed Stage-B source drifted: {name}")
        observed[name] = actual
    return observed


SOURCE_VERIFICATION = verify_stageb_sources()
value = str(STAGEB_SRC.resolve())
while value in sys.path:
    sys.path.remove(value)
sys.path.insert(0, value)

_DATA = importlib.import_module("c1b_stage_b.data")
_GATE = importlib.import_module("c1b_stage_b.gate")
_INPUTS = importlib.import_module("c1b_stage_b.inputs")
_TARGETS = importlib.import_module("c1b_stage_b.targets")
_TRAINING = importlib.import_module("c1b_stage_b.training")

for module, name in (
    (_DATA, "data.py"),
    (_GATE, "gate.py"),
    (_INPUTS, "inputs.py"),
    (_TARGETS, "targets.py"),
    (_TRAINING, "training.py"),
):
    expected = (STAGEB_SRC / "c1b_stage_b" / name).resolve()
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected:
        raise ImportError(f"sealed Stage-B module resolved outside canonical source: {module}")

StageBDataset = _DATA.StageBDataset
make_splits = _DATA.make_splits
require_stage_a_go = _GATE.require_stage_a_go
StageBDataPaths = _INPUTS.StageBDataPaths
load_stage_b_data = _INPUTS.load_stage_b_data
fit_grounding_transform = _TARGETS.fit_grounding_transform
logical_patient_batches = _TRAINING.logical_patient_batches
physical_patient_batches = _TRAINING.physical_patient_batches

for exported, module in (
    (StageBDataset, _DATA),
    (make_splits, _DATA),
    (require_stage_a_go, _GATE),
    (StageBDataPaths, _INPUTS),
    (load_stage_b_data, _INPUTS),
    (fit_grounding_transform, _TARGETS),
    (logical_patient_batches, _TRAINING),
    (physical_patient_batches, _TRAINING),
):
    if Path(inspect.getfile(inspect.unwrap(exported))).resolve() != Path(module.__file__).resolve():
        raise ImportError(f"sealed Stage-B object resolved outside canonical module: {exported}")


__all__ = [
    "SOURCE_VERIFICATION",
    "STAGEB_HASHES",
    "STAGEB_ROOT",
    "STAGEB_SRC",
    "StageBDataPaths",
    "StageBDataset",
    "fit_grounding_transform",
    "load_stage_b_data",
    "logical_patient_batches",
    "make_splits",
    "physical_patient_batches",
    "require_stage_a_go",
    "verify_stageb_sources",
]
