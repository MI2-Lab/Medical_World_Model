"""Contracts, small spatial readouts, and evaluation for Goal C.

The torch-dependent model module is intentionally imported lazily so that
contract validation, aggregate metrics, and privacy scans work in a CPU-only
or reporting-only environment.
"""

from .contracts import ExperimentContract, load_contract
from .metrics import classification_metrics

__all__ = ["ExperimentContract", "SpatialReadout", "classification_metrics", "load_contract"]


def __getattr__(name: str):
    if name == "SpatialReadout":
        from .models import SpatialReadout

        return SpatialReadout
    raise AttributeError(name)
