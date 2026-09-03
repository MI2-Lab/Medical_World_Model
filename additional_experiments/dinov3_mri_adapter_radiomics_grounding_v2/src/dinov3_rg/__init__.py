"""Outcome-blind DINOv3 MRI adapter and radiomics-grounding experiment."""

from .contracts import ARMS, FOLDS, SEEDS, VISITS, load_protocol

__all__ = ["ARMS", "FOLDS", "SEEDS", "VISITS", "MRIAdapterWorldModel", "load_protocol"]


def __getattr__(name: str):
    # The PyRadiomics Python 3.9 side environment intentionally has no torch.
    if name == "MRIAdapterWorldModel":
        from .model import MRIAdapterWorldModel

        return MRIAdapterWorldModel
    raise AttributeError(name)
