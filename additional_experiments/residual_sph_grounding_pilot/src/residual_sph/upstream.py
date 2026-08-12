"""Hash-locked imports from the tracked LOCAL3 implementation.

The later multiseed-confirmation tree is not part of this branch's Git
ancestry.  Its LOCAL3 arithmetic is identical to the tracked two-seed pilot,
so this experiment imports the tracked implementation and records exact
source hashes.  Private confirmed checkpoints remain runtime inputs only.
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
LOCAL_ROOT = REPO_ROOT / "additional_experiments" / "local_global_response_state_pilot"
LOCAL_SRC = LOCAL_ROOT / "src"

LOCAL_SOURCE_SHA256 = {
    "__init__.py": "0958d0c6530e249d2da6fa27ff1866ba717ccab31eea8c1402acac686d11504b",
    "contracts.py": "12298a28a5958cf75ced3412a07a49607afa53d4aed060dd844d1326b21d52a2",
    "pooling.py": "52ef08c85ed256a46c686c7f1afd1b66219735508f4a29cc0922fdb6096bb25c",
    "upstream.py": "8e537d514eabdd7f7c1d7c2234d6fc221610ea0ab5791e9e628e58c6f0e7a4de",
    "model.py": "09cc2959480bec953fe1fa7e92a53dbdf8be6c4d174151e00fef73e70929df76",
    "training.py": "f344f17fa722c4e942a3f5e42092bfaac06dabdcf6d672280384e804e6f0bd9e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    package = LOCAL_SRC / "lg_response_pilot"
    for name, expected in LOCAL_SOURCE_SHA256.items():
        path = package / name
        if not path.is_file():
            raise ImportError(f"tracked LOCAL source is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ImportError(
                f"tracked LOCAL source hash drifted for {name}: "
                f"expected {expected}, observed {actual}"
            )
        observed[name] = actual
    return observed


SOURCE_VERIFICATION = verify_local_sources()
source_value = str(LOCAL_SRC.resolve())
while source_value in sys.path:
    sys.path.remove(source_value)
sys.path.insert(0, source_value)

_PACKAGE = importlib.import_module("lg_response_pilot")
_MODEL = importlib.import_module("lg_response_pilot.model")
_TRAINING = importlib.import_module("lg_response_pilot.training")
_UPSTREAM = importlib.import_module("lg_response_pilot.upstream")

_EXPECTED = {
    _PACKAGE: (LOCAL_SRC / "lg_response_pilot" / "__init__.py").resolve(),
    _MODEL: (LOCAL_SRC / "lg_response_pilot" / "model.py").resolve(),
    _TRAINING: (LOCAL_SRC / "lg_response_pilot" / "training.py").resolve(),
    _UPSTREAM: (LOCAL_SRC / "lg_response_pilot" / "upstream.py").resolve(),
}
for module, expected in _EXPECTED.items():
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected:
        raise ImportError(f"LOCAL module resolved outside tracked source: {module.__name__}")
for value, expected in (
    (_MODEL.LocalGlobalResponseWorldModel, _EXPECTED[_MODEL]),
    (_MODEL.build_model, _EXPECTED[_MODEL]),
    (_UPSTREAM.DGRSObjective, _UPSTREAM.G3_SRC / "dgrs" / "training.py"),
):
    if Path(inspect.getfile(inspect.unwrap(value))).resolve() != Path(expected).resolve():
        raise ImportError(f"LOCAL object resolved outside its authenticated source: {value}")

LocalGlobalResponseWorldModel = _MODEL.LocalGlobalResponseWorldModel
build_local_model = _MODEL.build_model
local_tensor_state_sha256 = _MODEL.tensor_state_sha256
validate_local_model_contract = _MODEL.validate_model_contract
DGRSOutput = _UPSTREAM.DGRSOutput
DGRSObjective = _UPSTREAM.DGRSObjective

# These are re-exported by the hash-locked LOCAL trainer after it validates the
# sealed Stage-B implementation.
StageBTrainHyperparameters = _TRAINING.TrainHyperparameters
logical_patient_batches = _TRAINING.logical_patient_batches
physical_patient_batches = _TRAINING.physical_patient_batches


def source_contract() -> dict[str, Any]:
    return {
        "tracked_local_root": "additional_experiments/local_global_response_state_pilot",
        "source_sha256": dict(SOURCE_VERIFICATION),
        "confirmation_source_commit": "b4ec0c1473da513f2b19baa58d54c0fd5382e52f",
        "confirmation_tree_is_runtime_only_on_this_branch": True,
    }


__all__ = [
    "DGRSObjective",
    "DGRSOutput",
    "LOCAL_SOURCE_SHA256",
    "LocalGlobalResponseWorldModel",
    "SOURCE_VERIFICATION",
    "StageBTrainHyperparameters",
    "build_local_model",
    "local_tensor_state_sha256",
    "logical_patient_batches",
    "physical_patient_batches",
    "source_contract",
    "validate_local_model_contract",
    "verify_local_sources",
]
