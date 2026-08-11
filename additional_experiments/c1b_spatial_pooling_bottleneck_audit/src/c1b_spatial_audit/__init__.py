"""Frozen C1B spatial information / pooling bottleneck audit."""

from .pooling import (
    FINAL_CHANNELS,
    FINAL_STAGE_GEOMETRY,
    LOCAL_WINDOW_MM_XYZ,
    RESPONSE_DIM,
    S3_STAGE_GEOMETRY,
    StageGeometry,
    apply_frozen_response_projection,
    concatenate_local_global,
    expected_feature_shape,
    fixed_physical_local_weights,
    global_average_pool,
    receptive_field_occupancy,
    weighted_average_pool,
)

__all__ = [
    "FINAL_CHANNELS",
    "FINAL_STAGE_GEOMETRY",
    "LOCAL_WINDOW_MM_XYZ",
    "RESPONSE_DIM",
    "S3_STAGE_GEOMETRY",
    "StageGeometry",
    "apply_frozen_response_projection",
    "concatenate_local_global",
    "expected_feature_shape",
    "fixed_physical_local_weights",
    "global_average_pool",
    "receptive_field_occupancy",
    "weighted_average_pool",
]
