"""Stage B FTV-only 2x2 representation-sanity implementation."""

from .contracts import ARMS, ARM_SPECS, FOLDS, SEED_BASES, ArmSpec, arm_spec
from .gate import StageAGateError, require_stage_a_go

__all__ = [
    "ARMS",
    "ARM_SPECS",
    "FOLDS",
    "SEED_BASES",
    "ArmSpec",
    "StageAGateError",
    "arm_spec",
    "require_stage_a_go",
]
