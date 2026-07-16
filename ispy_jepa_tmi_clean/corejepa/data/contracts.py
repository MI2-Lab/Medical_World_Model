"""Named tensor dimensions used throughout the clean implementation."""

VISITS = ("T0", "T1", "T2", "T3")
N_VISITS = 4
N_TRANSITIONS = N_VISITS - 1
DCE8_CHANNELS = (
    "pre",
    "early",
    "late",
    "early_minus_pre",
    "late_minus_pre",
    "peak_relative_enhancement",
    "washout_relative_enhancement",
    "roi",
)
GEOMETRY_FEATURES = (
    "roi_volume_fraction",
    "bbox_size_z",
    "bbox_size_y",
    "bbox_size_x",
    "bbox_volume_fraction",
    "bbox_fill_fraction",
    "center_z",
    "center_y",
    "center_x",
)
TEMPORAL_CONDITION_FEATURES = (
    "target_T1",
    "target_T2",
    "target_T3",
    "observed_T0",
    "observed_T1",
    "observed_T2",
    "observed_T3",
)
RESPONSE_VECTOR_FEATURES = (
    "roi_volume_logratio_T0",
    "roi_volume_logratio_previous",
    "bbox_volume_logratio_T0",
    "bbox_volume_logratio_previous",
    "longest_diameter_logratio_T0",
    "longest_diameter_logratio_previous",
    "bbox_fill_delta_T0",
    "bbox_fill_delta_previous",
    "early_enhancement_delta_T0",
    "early_enhancement_delta_previous",
    "peak_enhancement_delta_T0",
    "peak_enhancement_delta_previous",
    "washout_delta_T0",
    "washout_delta_previous",
    "enhancement_auc_delta_T0",
    "enhancement_auc_delta_previous",
    "time_to_peak_delta_T0",
    "time_to_peak_delta_previous",
)
