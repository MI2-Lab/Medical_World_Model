"""Locked architecture, geometry, and transition-condition contracts for A1.

The encoder and LOCAL-overlap routines are deliberately loaded from their
completed upstream files.  Their source hashes are checked before import so a
future edit cannot silently change the preregistered A1 model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import operator
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]

G3_MODEL_PATH = (
    REPO_ROOT
    / "additional_experiments"
    / "g3_multiseed_generalization"
    / "src"
    / "dgrs"
    / "model.py"
)
AUDITED_POOLING_PATH = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_spatial_pooling_bottleneck_audit"
    / "src"
    / "c1b_spatial_audit"
    / "pooling.py"
)
G3_MODEL_SHA256 = "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9"
AUDITED_POOLING_SHA256 = (
    "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54"
)

C1B_INPUT_SHAPE_ZYX = (112, 176, 160)
C1B_SPACING_XYZ_MM = (0.9, 0.9, 2.0)
LOCAL_WINDOW_MM_XYZ = (64.0, 64.0, 64.0)
FINAL_STRIDE_ZYX = (8, 8, 8)
FINAL_CENTER_OFFSET_ZYX = (0.0, 0.0, 0.0)
IMAGE_CHANNELS = 7
VISITS = 4
TRANSITIONS = 3
FINAL_CHANNELS = 128
TOKEN_DIM = 128
RESPONSE_DIM = 192
FORMAL_LOCAL_TOKEN_COUNT = 500
FORMAL_MASKED_TOKEN_COUNT = 250
MASK_RATIO = 0.5
POSITION_NORMALIZATION_MM = 32.0

PREDICTOR_BLOCKS = 4
PREDICTOR_HEADS = 8
PREDICTOR_FF_DIM = 512
PREDICTOR_DROPOUT = 0.1
ARM_EMBEDDING_DIM = 16

FIXED_ARM_VOCAB = (
    "ISPY1_NACT",
    "Paclitaxel",
    "Paclitaxel + ABT 888 + Carboplatin",
    "Paclitaxel + AMG 386",
    "Paclitaxel + AMG 386 + Trastuzumab",
    "Paclitaxel + Ganetespib",
    "Paclitaxel + Ganitumab",
    "Paclitaxel + MK-2206",
    "Paclitaxel + MK-2206 + Trastuzumab",
    "Paclitaxel + Neratinib",
    "Paclitaxel + Pembrolizumab",
    "Paclitaxel + Pertuzumab + Trastuzumab",
    "Paclitaxel + Trastuzumab",
    "T-DM1 + Pertuzumab",
)
CLINICAL_FEATURES = ("HR", "HER2", "MP", "age_z", "age_missing")
TEMPORAL_FEATURES = (
    "target_T1",
    "target_T2",
    "target_T3",
    "observed_T0",
    "observed_T1",
    "observed_T2",
    "observed_T3",
)
NOMINAL_TEMPORAL_BITS = (
    (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
)
NOMINAL_DELTA_T = (1.0, 1.0, 1.0)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without changing the source file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_locked_module(name: str, path: Path, expected_sha256: str) -> ModuleType:
    if not path.is_file():
        raise ImportError(f"required upstream source is missing: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ImportError(
            f"hash-locked upstream source drifted at {path}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(str(getattr(existing, "__file__", ""))).resolve() != path.resolve():
            raise ImportError(f"module name {name!r} already resolves to another file")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not construct an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules during execution.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


G3_MODEL_MODULE = _hash_locked_module(
    "_patch_token_wm_hashlocked_g3_model", G3_MODEL_PATH, G3_MODEL_SHA256
)
AUDITED_POOLING_MODULE = _hash_locked_module(
    "_patch_token_wm_hashlocked_c1b_pooling",
    AUDITED_POOLING_PATH,
    AUDITED_POOLING_SHA256,
)

# This is the class object executed directly from the hash-locked G3 source;
# no encoder block is copied or locally reimplemented.
SpatialVisitEncoder3D = G3_MODEL_MODULE.SpatialVisitEncoder3D
audited_expected_feature_shape = AUDITED_POOLING_MODULE.expected_feature_shape
audited_fixed_physical_local_weights = (
    AUDITED_POOLING_MODULE.fixed_physical_local_weights
)
audited_weighted_average_pool = AUDITED_POOLING_MODULE.weighted_average_pool


def _positive_int_triplet(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain three ZYX values")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain positive integers")
        try:
            parsed = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain positive integers") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must contain positive integers")
        output.append(parsed)
    return tuple(output)  # type: ignore[return-value]


def validate_geometry_values(
    input_shape_zyx: Sequence[int], spacing_xyz_mm: Sequence[float]
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    shape = _positive_int_triplet(input_shape_zyx, "input_shape_zyx")
    if isinstance(spacing_xyz_mm, (str, bytes)) or len(spacing_xyz_mm) != 3:
        raise ValueError("spacing_xyz_mm must contain three XYZ values")
    spacing = tuple(float(value) for value in spacing_xyz_mm)
    if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in spacing):
        raise ValueError("spacing_xyz_mm must contain finite positive values")
    return shape, spacing  # type: ignore[return-value]


def nominal_temporal_bits(
    batch_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the frozen seven prior nominal bits for T0->T1, T1->T2, T2->T3."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    base = torch.tensor(NOMINAL_TEMPORAL_BITS, device=device, dtype=dtype)
    return base.unsqueeze(0).expand(int(batch_size), -1, -1).clone()


@dataclass(frozen=True)
class TransitionCondition:
    """The complete pCR-free predictor condition.

    ``clinical`` is ordered as HR, HER2, MP, age_z, age_missing.  This object is
    consumed only by the transition predictor; MRI token encoders have no
    condition argument.
    """

    arm_index: torch.Tensor
    clinical: torch.Tensor
    temporal_bits: torch.Tensor
    delta_t: torch.Tensor

    @classmethod
    def nominal(
        cls, arm_index: torch.Tensor, clinical: torch.Tensor
    ) -> "TransitionCondition":
        if not isinstance(arm_index, torch.Tensor) or arm_index.ndim != 1:
            raise ValueError("arm_index must have shape [B]")
        batch = int(arm_index.shape[0])
        device = (
            clinical.device if isinstance(clinical, torch.Tensor) else arm_index.device
        )
        dtype = clinical.dtype if isinstance(clinical, torch.Tensor) else torch.float32
        return cls(
            arm_index=arm_index,
            clinical=clinical,
            temporal_bits=nominal_temporal_bits(batch, device=device, dtype=dtype),
            delta_t=torch.ones((batch, TRANSITIONS), device=device, dtype=dtype),
        )

    def validate(self, batch_size: int, device: torch.device) -> None:
        """Fail closed on vocabulary, shape, nominal-time, or device drift."""

        tensors = {
            "arm_index": self.arm_index,
            "clinical": self.clinical,
            "temporal_bits": self.temporal_bits,
            "delta_t": self.delta_t,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("all transition-condition fields must be torch tensors")
        expected = {
            "arm_index": (batch_size,),
            "clinical": (batch_size, len(CLINICAL_FEATURES)),
            "temporal_bits": (batch_size, TRANSITIONS, len(TEMPORAL_FEATURES)),
            "delta_t": (batch_size, TRANSITIONS),
        }
        for name, value in tensors.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(
                    f"{name} must have shape {expected[name]}, got {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(f"{name} and MRI image must be on the same device")
        if self.arm_index.dtype == torch.bool or self.arm_index.dtype.is_floating_point:
            raise TypeError("arm_index must use an integer dtype")
        if bool((self.arm_index < 0).any()) or bool(
            (self.arm_index >= len(FIXED_ARM_VOCAB)).any()
        ):
            raise ValueError("arm_index escaped the fixed 14-arm vocabulary")
        for name, value in (
            ("clinical", self.clinical),
            ("temporal_bits", self.temporal_bits),
            ("delta_t", self.delta_t),
        ):
            if not value.dtype.is_floating_point or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must contain finite floating values")
        if bool(((self.clinical[:, -1] != 0) & (self.clinical[:, -1] != 1)).any()):
            raise ValueError("age_missing must be an exact 0/1 indicator")
        expected_bits = nominal_temporal_bits(
            batch_size, device=device, dtype=self.temporal_bits.dtype
        )
        if not torch.equal(self.temporal_bits, expected_bits):
            raise ValueError(
                "temporal_bits differ from the seven locked nominal prior bits"
            )
        if not torch.equal(self.delta_t, torch.ones_like(self.delta_t)):
            raise ValueError("delta_t is locked to nominal adjacent interval 1.0")


def source_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "g3_model_path": str(G3_MODEL_PATH.relative_to(REPO_ROOT)),
        "g3_model_sha256": G3_MODEL_SHA256,
        "encoder_class": "SpatialVisitEncoder3D",
        "audited_pooling_path": str(AUDITED_POOLING_PATH.relative_to(REPO_ROOT)),
        "audited_pooling_sha256": AUDITED_POOLING_SHA256,
    }


__all__ = [
    "ARM_EMBEDDING_DIM",
    "AUDITED_POOLING_PATH",
    "AUDITED_POOLING_SHA256",
    "C1B_INPUT_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "CLINICAL_FEATURES",
    "EXPERIMENT_ROOT",
    "FINAL_CENTER_OFFSET_ZYX",
    "FINAL_CHANNELS",
    "FINAL_STRIDE_ZYX",
    "FIXED_ARM_VOCAB",
    "FORMAL_LOCAL_TOKEN_COUNT",
    "FORMAL_MASKED_TOKEN_COUNT",
    "G3_MODEL_PATH",
    "G3_MODEL_SHA256",
    "IMAGE_CHANNELS",
    "LOCAL_WINDOW_MM_XYZ",
    "MASK_RATIO",
    "NOMINAL_DELTA_T",
    "NOMINAL_TEMPORAL_BITS",
    "POSITION_NORMALIZATION_MM",
    "PREDICTOR_BLOCKS",
    "PREDICTOR_DROPOUT",
    "PREDICTOR_FF_DIM",
    "PREDICTOR_HEADS",
    "REPO_ROOT",
    "RESPONSE_DIM",
    "SpatialVisitEncoder3D",
    "TEMPORAL_FEATURES",
    "TOKEN_DIM",
    "TRANSITIONS",
    "TransitionCondition",
    "VISITS",
    "audited_expected_feature_shape",
    "audited_fixed_physical_local_weights",
    "audited_weighted_average_pool",
    "file_sha256",
    "nominal_temporal_bits",
    "source_contract",
    "validate_geometry_values",
]
