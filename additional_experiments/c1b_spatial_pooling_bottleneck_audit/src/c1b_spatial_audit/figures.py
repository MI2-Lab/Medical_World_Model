"""Deterministic, privacy-preserving figures for the frozen spatial audit.

The public renderer consumes the six aggregate public audit tables, the public
training-budget table, one private geometry/support sidecar, and one private
aggregate activation volume.  Identifier-bearing rows and raw activations are
never passed to a plotting function.  This module performs no probe fitting,
model training, pooling selection, or gate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib


# The formal job is headless.  Select the backend before importing pyplot so a
# workstation display configuration cannot alter execution.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np
import pandas as pd

from .analysis import (
    FTV_TABLE_COLUMNS,
    NUISANCE_TARGETS,
    TABLE1_COLUMNS,
    TABLE4_COLUMNS,
    TABLE5_COLUMNS,
    TABLE6_COLUMNS,
)
from .contracts import (
    ARMS,
    FOLDS,
    SEEDS,
    TIMEPOINTS,
    TRANSITIONS,
    cell_key,
    checkpoint_path,
    file_sha256,
    relative,
)
from .sidecars import SIDECAR_KEYS


FIGURE_DPI = 200
FIGURE_FILENAMES = (
    "01_feature_map_pooling_schematic.png",
    "02_pooling_weight_illustration.png",
    "03_static_macro_spearman.png",
    "04_static_natural_r2.png",
    "05_delta_macro_spearman.png",
    "06_legacy_deficit_recovery.png",
    "07_oracle_vs_local_recovery.png",
    "08_padding_valid_source_decodability.png",
    "09_ftv_vs_nuisance_information.png",
    "10_occupancy_n1_l1_degradation.png",
    "11_selected_epoch_training_budget.png",
    "12_representative_activation_montage.png",
)

TABLE7_COLUMNS = (
    "seed",
    "arm",
    "fold",
    "selected_epoch",
    "observed_max_epoch",
    "configured_max_epoch",
    "hit_configured_max_epoch",
    "selected_in_last_two_observed_epochs",
    "selected_validation_state_loss",
    "final_validation_state_loss",
    "final_minus_selected_state_loss",
    "last_three_normalized_validation_state_slope",
    "early_stopping_reason",
    "selection_mode",
    "optimization_safety_pass",
    "history_sha256",
    "selection_sha256",
)

ACTIVATION_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "activation_mean_zyx",
        "selected_patient_count",
        "visits_per_patient",
        "source_row_count",
        "channel_count",
        "feature_shape_zyx",
        "normalization",
        "selection_rule",
        "seed_base",
        "arm",
        "fold",
        "stage",
        "checkpoint_sha256",
        "training_performed",
        "outcomes_used",
    }
)
ACTIVATION_SELECTION_COUNT = 16
ACTIVATION_VISIT_COUNT = 4
ACTIVATION_CHANNEL_COUNT = 128
ACTIVATION_SHAPE_ZYX = (14, 22, 20)
ACTIVATION_NORMALIZATION = (
    "per_patient_visit_abs_max_then_mean_over_channels_patients_visits"
)
ACTIVATION_SELECTION_RULE = "16_smallest_sha256_utf8_patient_id"

_FINAL_POOLINGS = ("P0", "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE")
_S3_POOLINGS = ("P0", "PLOCAL", "PORACLE")
_SECONDARY_POOLING = "PLOCAL+PVALID_SECONDARY"
_NUISANCE_POOLINGS = ("P0", "PVALID", "PLOCAL")
_DOWNSAMPLING_BINS = ("<=1.5", "(1.5,2]", ">2")
_PRIMARY_SCOPE = "primary_measurement_valid"
_ENDPOINT_AGGREGATION = "pooled_outer_test_folds"
_MACRO_AGGREGATION = "mean_of_pooled_endpoint_metrics"
_ARM_COLORS = {
    "L1": "#3769a6",
    "L3": "#65a0d4",
    "N1": "#d65f4a",
    "N3": "#e6a24a",
}
_POOLING_COLORS = {
    "P0": "#6c757d",
    "PVALID": "#2a9d8f",
    "PLOCAL": "#457b9d",
    "PLOCAL+GLOBAL": "#7353ba",
    "PORACLE": "#e76f51",
    _SECONDARY_POOLING: "#8a8f98",
}
_POOLING_LABELS = {
    "P0": "P0",
    "PVALID": "PVALID",
    "PLOCAL": "PLOCAL",
    "PLOCAL+GLOBAL": "PLOCAL+GLOBAL",
    "PORACLE": "PORACLE",
    _SECONDARY_POOLING: "PLOCAL+PVALID*",
}


@dataclass(frozen=True)
class FigureInputPaths:
    table1: Path
    table2: Path
    table3: Path
    table4: Path
    table5: Path
    table6: Path
    table7: Path
    sidecar: Path
    activation_aggregate: Path


@dataclass(frozen=True)
class FigureData:
    table1: pd.DataFrame
    table2: pd.DataFrame
    table3: pd.DataFrame
    table4: pd.DataFrame
    table5: pd.DataFrame
    table6: pd.DataFrame
    table7: pd.DataFrame
    pooling_maps: Mapping[str, np.ndarray]
    activation_mean_zyx: np.ndarray


@dataclass(frozen=True)
class ActivationAggregate:
    activation_mean_zyx: np.ndarray
    checkpoint_sha256: str


def _scalar(array: np.ndarray, *, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar array")
    return value.item()


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"{label} contains a non-boolean value: {value!r}")


def _is_public_missing(value: Any) -> bool:
    """Return whether a keep-default-NA-disabled public CSV cell is empty."""

    return bool(pd.isna(value) or (isinstance(value, str) and value == ""))


def _available_feature_dim(stage: str, pooling: str) -> int:
    if stage == "s3":
        return 64
    if pooling in {"PLOCAL+GLOBAL", _SECONDARY_POOLING}:
        return 384
    return 192


def _ftv_analysis_role(stage: str, pooling: str) -> str:
    if pooling == _SECONDARY_POOLING:
        return "secondary_sensitivity"
    return "conditional_s3" if stage == "s3" else "primary"


def _natural_aggregation(endpoint: str) -> str:
    return _MACRO_AGGREGATION if endpoint == "macro" else _ENDPOINT_AGGREGATION


def _require_exact_schema(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    if tuple(frame.columns) != tuple(columns):
        missing = sorted(set(columns) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(columns))
        raise ValueError(
            f"{label} schema drift: missing={missing}, extra={extra}, "
            "or column order changed"
        )
    forbidden = {"patient_id", "cache_path", "ftv_mask_nifti"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"{label} contains private columns: {sorted(forbidden)}")


def _require_finite(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains a missing/non-finite {column}")


def _load_csv(path: Path, columns: Sequence[str], *, label: str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # ``NA`` is an intentional public availability label, not a missing value.
    # Empty numeric cells are coerced explicitly by the validators below.
    frame = pd.read_csv(source, keep_default_na=False)
    _require_exact_schema(frame, columns, label=label)
    return frame


def _validate_table1(frame: pd.DataFrame) -> None:
    _require_exact_schema(frame, TABLE1_COLUMNS, label="Table 1")
    identities = set(zip(frame["stage"].astype(str), frame["input_contract"].astype(str)))
    expected = {
        (stage, contract)
        for stage in ("final", "s3")
        for contract in ("legacy", "c1b")
    }
    if len(frame) != 4 or identities != expected:
        raise ValueError("Table 1 must be the exact final/S3 x legacy/C1B contract")
    if frame.duplicated(["stage", "input_contract"]).any():
        raise ValueError("Table 1 contains duplicate feature contracts")
    expected_geometry = {
        ("final", "legacy"): (
            "primary",
            "32x96x96",
            128,
            "4x12x12",
            8,
            47,
        ),
        ("final", "c1b"): (
            "primary",
            "112x176x160",
            128,
            "14x22x20",
            8,
            47,
        ),
        ("s3", "legacy"): (
            "conditional_secondary",
            "32x96x96",
            64,
            "8x24x24",
            4,
            23,
        ),
        ("s3", "c1b"): (
            "conditional_secondary",
            "112x176x160",
            64,
            "28x44x40",
            4,
            23,
        ),
    }
    for row in frame.itertuples(index=False):
        key = (str(row.stage), str(row.input_contract))
        observed = (
            str(row.analysis_role),
            str(row.input_shape_zyx),
            int(row.feature_channels),
            str(row.feature_shape_zyx),
            int(row.jump_input_voxels),
            int(row.theoretical_receptive_field_input_voxels),
        )
        if observed != expected_geometry[key]:
            raise ValueError(f"Table 1 frozen geometry drift at {key}: {observed}")
        if int(row.center_offset_input_voxels) != 0:
            raise ValueError("Table 1 feature-center offset drifted")
        if str(row.local_window_mm_xyz) != "64x64x64":
            raise ValueError("Table 1 fixed physical local-window contract drifted")
    _require_finite(
        frame,
        (
            "jump_x_mm_median",
            "jump_x_mm_q25",
            "jump_x_mm_q75",
            "jump_y_mm_median",
            "jump_y_mm_q25",
            "jump_y_mm_q75",
            "jump_z_mm_median",
            "jump_z_mm_q25",
            "jump_z_mm_q75",
        ),
        label="Table 1",
    )


def _expected_ftv_identities(
    *, endpoints: Sequence[str], stages: set[str], secondary_present: bool
) -> set[tuple[str, int, str, str, str]]:
    expected: set[tuple[str, int, str, str, str]] = set()
    for seed in SEEDS:
        for arm in ARMS:
            for pooling in _FINAL_POOLINGS:
                for endpoint in endpoints:
                    expected.add(("final", seed, arm, pooling, endpoint))
            if secondary_present and arm in {"N1", "N3"}:
                for endpoint in endpoints:
                    expected.add(("final", seed, arm, _SECONDARY_POOLING, endpoint))
    if "s3" in stages:
        for seed in SEEDS:
            for arm in ARMS:
                for pooling in _S3_POOLINGS:
                    for endpoint in endpoints:
                        expected.add(("s3", seed, arm, pooling, endpoint))
    return expected


def _validate_ftv_table(
    frame: pd.DataFrame, *, endpoints: Sequence[str], label: str
) -> tuple[frozenset[str], bool]:
    _require_exact_schema(frame, FTV_TABLE_COLUMNS, label=label)
    stages = set(frame["stage"].dropna().astype(str))
    if stages not in ({"final"}, {"final", "s3"}):
        raise ValueError(f"{label} has an invalid final/S3 stage set: {sorted(stages)}")
    secondary_rows = frame["pooling"].astype(str).eq(_SECONDARY_POOLING)
    secondary_present = bool(secondary_rows.any())
    identities = set(
        zip(
            frame["stage"].astype(str),
            frame["seed_base"].astype(int),
            frame["arm"].astype(str),
            frame["pooling"].astype(str),
            frame["endpoint"].astype(str),
        )
    )
    expected = _expected_ftv_identities(
        endpoints=endpoints, stages=stages, secondary_present=secondary_present
    )
    if identities != expected or len(frame) != len(expected):
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(
            f"{label} rectangular matrix drift: missing={missing[:3]}, extra={extra[:3]}"
        )
    if frame.duplicated(["stage", "seed_base", "arm", "pooling", "endpoint"]).any():
        raise ValueError(f"{label} contains duplicate aggregate cells")
    for row in frame.itertuples(index=False):
        stage = str(row.stage)
        pooling = str(row.pooling)
        legacy = str(row.arm) in {"L1", "L3"}
        unavailable = legacy and (
            (stage == "final" and pooling in {"PVALID", "PORACLE"})
            or (stage == "s3" and pooling == "PORACLE")
        )
        expected_availability = "NA" if unavailable else "AVAILABLE"
        if str(row.availability) != expected_availability:
            raise ValueError(
                f"{label} availability drift at "
                f"{row.stage}/{row.seed_base}/{row.arm}/{row.pooling}/{row.endpoint}"
            )
        if str(row.analysis_role) != _ftv_analysis_role(stage, pooling):
            raise ValueError(
                f"{label} analysis-role drift at "
                f"{row.stage}/{row.seed_base}/{row.arm}/{row.pooling}/{row.endpoint}"
            )
        if str(row.analysis_scope) != _PRIMARY_SCOPE:
            raise ValueError(f"{label} analysis scope must remain {_PRIMARY_SCOPE}")
        if unavailable:
            if not _is_public_missing(row.feature_dim):
                raise ValueError(f"{label} unavailable row claims a feature dimension")
            if not _is_public_missing(row.aggregation):
                raise ValueError(f"{label} unavailable row claims an aggregation")
            if _is_public_missing(row.status_reason):
                raise ValueError(f"{label} unavailable row has no explicit NA reason")
        else:
            dimension = pd.to_numeric(
                pd.Series([row.feature_dim]), errors="coerce"
            ).iloc[0]
            expected_dimension = _available_feature_dim(stage, pooling)
            if (
                not np.isfinite(float(dimension))
                or float(dimension) != float(expected_dimension)
            ):
                raise ValueError(
                    f"{label} feature_dim drift at {stage}/{row.arm}/{pooling}: "
                    f"expected {expected_dimension}, got {row.feature_dim!r}"
                )
            expected_aggregation = _natural_aggregation(str(row.endpoint))
            if str(row.aggregation) != expected_aggregation:
                raise ValueError(
                    f"{label} natural metrics must use {expected_aggregation}"
                )
            if not _is_public_missing(row.status_reason):
                raise ValueError(f"{label} available row has a nonempty status reason")
    plotted = frame.loc[
        frame["endpoint"].astype(str).eq("macro")
        & frame["availability"].astype(str).eq("AVAILABLE")
    ]
    _require_finite(plotted, ("spearman", "natural_r2"), label=f"{label} macro rows")
    return frozenset(stages), secondary_present


def _validate_table4(frame: pd.DataFrame, *, stages: set[str]) -> None:
    _require_exact_schema(frame, TABLE4_COLUMNS, label="Table 4")
    expected: set[tuple[str, int, str, str]] = set()
    for stage in stages:
        poolings = _FINAL_POOLINGS if stage == "final" else _S3_POOLINGS
        for seed in SEEDS:
            for arm in ("N1", "N3"):
                for pooling in poolings:
                    expected.add((stage, seed, arm, pooling))
    identities = set(
        zip(
            frame["stage"].astype(str),
            frame["seed_base"].astype(int),
            frame["new_arm"].astype(str),
            frame["pooling"].astype(str),
        )
    )
    if identities != expected or len(frame) != len(expected):
        raise ValueError("Table 4 is not the exact final/conditional-S3 recovery matrix")
    if frame.duplicated(["stage", "seed_base", "new_arm", "pooling"]).any():
        raise ValueError("Table 4 contains duplicate recovery cells")
    _require_finite(
        frame,
        (
            "legacy_p0_spearman",
            "new_p0_spearman",
            "legacy_deficit",
            "pooling_spearman",
            "absolute_gain_vs_new_p0",
        ),
        label="Table 4",
    )
    for index, row in frame.iterrows():
        expected_role = "primary" if str(row["new_arm"]) == "N1" else "secondary_replication"
        if str(row["analysis_role"]) != expected_role:
            raise ValueError(f"Table 4 analysis-role drift at row {index}")
        defined = _strict_bool(row["recovery_defined"], label="Table 4 recovery_defined")
        ratio = pd.to_numeric(pd.Series([row["recovery_ratio"]]), errors="coerce").iloc[0]
        if defined and not np.isfinite(float(ratio)):
            raise ValueError(f"Table 4 has missing defined recovery ratio at row {index}")
        if not defined and pd.notna(ratio) and np.isfinite(float(ratio)):
            raise ValueError(f"Table 4 has numeric undefined recovery ratio at row {index}")


def _validate_table5(frame: pd.DataFrame) -> None:
    _require_exact_schema(frame, TABLE5_COLUMNS, label="Table 5")
    endpoints = (*TIMEPOINTS, "macro")
    expected = {
        ("final", seed, arm, pooling, target, endpoint)
        for seed in SEEDS
        for arm in ARMS
        for pooling in _NUISANCE_POOLINGS
        for target in NUISANCE_TARGETS
        for endpoint in endpoints
    }
    identities = set(
        zip(
            frame["stage"].astype(str),
            frame["seed_base"].astype(int),
            frame["arm"].astype(str),
            frame["pooling"].astype(str),
            frame["target_name"].astype(str),
            frame["endpoint"].astype(str),
        )
    )
    if identities != expected or len(frame) != len(expected):
        raise ValueError("Table 5 is not the exact nuisance target matrix")
    if frame.duplicated(
        ["stage", "seed_base", "arm", "pooling", "target_name", "endpoint"]
    ).any():
        raise ValueError("Table 5 contains duplicate nuisance cells")
    for row in frame.itertuples(index=False):
        unavailable = str(row.arm) in {"L1", "L3"} and str(row.pooling) == "PVALID"
        if str(row.availability) != ("NA" if unavailable else "AVAILABLE"):
            raise ValueError("Table 5 legacy PVALID availability drifted")
        if unavailable:
            if not _is_public_missing(row.feature_dim):
                raise ValueError("Table 5 unavailable row claims a feature dimension")
            if not _is_public_missing(row.aggregation):
                raise ValueError("Table 5 unavailable row claims an aggregation")
            if _is_public_missing(row.status_reason):
                raise ValueError("Table 5 unavailable row has no explicit NA reason")
        else:
            dimension = pd.to_numeric(
                pd.Series([row.feature_dim]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(float(dimension)) or float(dimension) != 192.0:
                raise ValueError("Table 5 final nuisance feature_dim must be 192")
            expected_aggregation = _natural_aggregation(str(row.endpoint))
            if str(row.aggregation) != expected_aggregation:
                raise ValueError(
                    f"Table 5 natural metrics must use {expected_aggregation}"
                )
            if not _is_public_missing(row.status_reason):
                raise ValueError("Table 5 available row has a nonempty status reason")
    plotted = frame.loc[
        frame["endpoint"].astype(str).eq("macro")
        & frame["availability"].astype(str).eq("AVAILABLE")
    ]
    _require_finite(plotted, ("spearman", "natural_r2"), label="Table 5 macro rows")


def _validate_table6(frame: pd.DataFrame) -> None:
    _require_exact_schema(frame, TABLE6_COLUMNS, label="Table 6")
    expected: set[tuple[str, int, str, str]] = set()
    for seed in SEEDS:
        for endpoint in TIMEPOINTS:
            expected.update(
                ("occupancy_quartile", seed, endpoint, quartile)
                for quartile in ("Q1", "Q2", "Q3", "Q4")
            )
            expected.add(("occupancy_correlation", seed, endpoint, "ALL"))
            expected.update(
                ("downsampling_bin", seed, endpoint, label)
                for label in _DOWNSAMPLING_BINS
            )
            expected.add(("downsampling_correlation", seed, endpoint, "ALL"))
    identities = set(
        zip(
            frame["analysis"].astype(str),
            frame["seed_base"].astype(int),
            frame["endpoint"].astype(str),
            frame["stratum"].astype(str),
        )
    )
    if identities != expected or len(frame) != len(expected):
        raise ValueError("Table 6 is not the exact occupancy/downsampling matrix")
    if frame.duplicated(["analysis", "seed_base", "endpoint", "stratum"]).any():
        raise ValueError("Table 6 contains duplicate diagnostic cells")
    occupancy = frame.loc[frame["analysis"].astype(str).eq("occupancy_quartile")]
    _require_finite(
        occupancy,
        (
            "n",
            "l1_mae",
            "n1_mae",
            "n1_minus_l1_mae",
            "mean_paired_abs_error_difference",
        ),
        label="Table 6 occupancy rows",
    )
    occupancy_n = pd.to_numeric(occupancy["n"], errors="coerce")
    if (
        (occupancy_n <= 0).any()
        or not np.equal(occupancy_n, np.floor(occupancy_n)).all()
    ):
        raise ValueError("Table 6 contains an empty occupancy quartile")
    # A rank correlation is undefined for a singleton stratum.  The formal
    # pooled occupancy qcut legitimately yields T0/Q1 n=1, so require an
    # explicit NA there while remaining fail-closed for every n>=2 row.
    singleton = occupancy_n.lt(2)
    for column in ("l1_spearman", "n1_spearman", "n1_minus_l1_spearman"):
        values = pd.to_numeric(occupancy[column], errors="coerce")
        if values.loc[~singleton].isna().any():
            raise ValueError(
                f"Table 6 occupancy rows contain a missing/non-finite {column}"
            )
        if values.loc[singleton].notna().any():
            raise ValueError(
                f"Table 6 singleton occupancy rows must mark {column} unavailable"
            )
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError("Table 6 contains an infinite value")


def _validate_table7(frame: pd.DataFrame) -> None:
    _require_exact_schema(frame, TABLE7_COLUMNS, label="Table 7")
    expected = {(seed, arm, fold) for seed in SEEDS for arm in ARMS for fold in FOLDS}
    identities = set(
        zip(frame["seed"].astype(int), frame["arm"].astype(str), frame["fold"].astype(int))
    )
    if identities != expected or len(frame) != 40:
        raise ValueError("Table 7 is not the exact frozen 40-cell matrix")
    _require_finite(
        frame,
        (
            "selected_epoch",
            "observed_max_epoch",
            "configured_max_epoch",
            "selected_validation_state_loss",
            "final_validation_state_loss",
            "final_minus_selected_state_loss",
            "last_three_normalized_validation_state_slope",
        ),
        label="Table 7",
    )
    selected = pd.to_numeric(frame["selected_epoch"]).to_numpy(dtype=np.int64)
    observed = pd.to_numeric(frame["observed_max_epoch"]).to_numpy(dtype=np.int64)
    configured = pd.to_numeric(frame["configured_max_epoch"]).to_numpy(dtype=np.int64)
    if not np.equal(configured, 12).all() or np.any(selected < 1) or np.any(selected > observed):
        raise ValueError("Table 7 epoch contract drifted")
    for column in (
        "hit_configured_max_epoch",
        "selected_in_last_two_observed_epochs",
        "optimization_safety_pass",
    ):
        frame[column].map(lambda value: _strict_bool(value, label=f"Table 7 {column}"))
    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    for column in ("history_sha256", "selection_sha256"):
        if not frame[column].astype(str).map(lambda value: bool(hash_pattern.fullmatch(value))).all():
            raise ValueError(f"Table 7 contains an invalid {column}")


def load_pooling_weight_aggregates(path: str | Path) -> dict[str, np.ndarray]:
    """Return four identifier-free aggregate maps and discard all source rows."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or not source.name.endswith(".private.npz"):
        raise FileNotFoundError(f"private audit sidecar is absent: {source}")
    if source.stat().st_mode & 0o077:
        raise PermissionError("private audit sidecar must be owner-only")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(SIDECAR_KEYS):
            raise ValueError("audit sidecar key inventory drifted")
        patient_ids = np.asarray(archive["patient_id"]).astype(str)
        if patient_ids.shape != (808,) or len(set(patient_ids.tolist())) != 808:
            raise ValueError("audit sidecar does not contain the exact 808-patient set")
        valid = np.asarray(archive["c1b_valid_weight_final"])
        oracle = np.asarray(archive["c1b_oracle_weight_final"])
        oracle_valid = np.asarray(archive["c1b_oracle_valid"])
        local = np.asarray(archive["c1b_local_weight_final"])
        legacy_local = np.asarray(archive["legacy_local_weight_final"])
    expected = (808, 4, 14, 22, 20)
    for name, array in (("C1B valid", valid), ("C1B oracle", oracle)):
        if array.dtype != np.float32 or array.shape != expected:
            raise ValueError(f"{name} weight shape/dtype drifted")
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{name} weights escaped finite [0,1]")
    if oracle_valid.dtype != np.bool_ or oracle_valid.shape != (808, 4):
        raise ValueError("C1B oracle-valid mask shape/dtype drifted")
    if int(oracle_valid.sum()) != 1500:
        raise ValueError("C1B oracle-valid population drifted from 1500 visits")
    if np.any(oracle[~oracle_valid] != 0):
        raise ValueError("invalid oracle rows contain nonzero support")
    if local.dtype != np.float32 or local.shape != (14, 22, 20):
        raise ValueError("C1B local weights drifted")
    if legacy_local.dtype != np.float32 or legacy_local.shape != (808, 4, 4, 12, 12):
        raise ValueError("legacy local weights drifted")
    for name, array in (("C1B local", local), ("legacy local", legacy_local)):
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{name} weights escaped finite [0,1]")
    if np.any(valid.reshape(808, 4, -1).sum(axis=-1) <= 0):
        raise ValueError("C1B valid-source weight has empty support")
    if np.any(oracle[oracle_valid].reshape(1500, -1).sum(axis=-1) <= 0):
        raise ValueError("C1B oracle weight has empty formal support")
    if float(local.sum(dtype=np.float64)) <= 0:
        raise ValueError("C1B local weight has empty support")

    # Only these four small aggregates leave this function.  Patient IDs and
    # per-patient masks go out of scope before plotting begins.  PORACLE remains
    # a non-deployable diagnostic because its aggregate uses lesion support.
    return {
        "P0": np.ones((14, 22, 20), dtype=np.float32),
        "PVALID": valid.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
        "PLOCAL": local.astype(np.float32, copy=True),
        "PORACLE": oracle[oracle_valid].mean(axis=0, dtype=np.float64).astype(np.float32),
    }


def _fixed_activation_checkpoint_binding() -> tuple[Path, str]:
    """Resolve and live-verify the preregistered 2026/N1/fold-0 checkpoint.

    The path is used only inside the private build.  Public rendering consumes
    only the expected digest, so neither an absolute nor a repository-relative
    checkpoint path is embedded in the aggregate or a PNG.
    """

    from .runtime import verify_preregistration

    lock = verify_preregistration()
    checkpoint = checkpoint_path(2026, "N1", 0).resolve()
    locked = lock.get("selected_checkpoints", {}).get(cell_key(2026, "N1", 0))
    if not isinstance(locked, Mapping):
        raise ValueError("preregistration is missing the fixed activation checkpoint")
    digest = str(locked.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("fixed activation checkpoint lock digest is malformed")
    if locked.get("path") != relative(checkpoint):
        raise ValueError("fixed activation checkpoint path differs from preregistration")
    if not checkpoint.is_file():
        raise FileNotFoundError("fixed activation checkpoint is absent")
    locked_size = int(locked.get("size_bytes", -1))
    if checkpoint.stat().st_size != locked_size:
        raise ValueError("fixed activation checkpoint size differs from preregistration")
    if file_sha256(checkpoint) != digest:
        raise ValueError("fixed activation checkpoint SHA-256 differs from preregistration")
    return checkpoint, digest


def select_patient_ids_by_sha256(
    patient_ids: Sequence[str], *, count: int = ACTIVATION_SELECTION_COUNT
) -> tuple[str, ...]:
    """Outcome-free deterministic selection by SHA256(raw UTF-8 patient ID)."""

    values = tuple(str(value) for value in patient_ids)
    if len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError("activation selection requires unique nonempty patient IDs")
    if int(count) <= 0 or int(count) > len(values):
        raise ValueError("activation selection count is outside the population")
    ranked = sorted(
        values,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).digest(), value),
    )
    return tuple(ranked[: int(count)])


def aggregate_normalized_abs_activations(activations: np.ndarray) -> np.ndarray:
    """Aggregate ``[patient,visit,channel,Z,Y,X]`` without retaining rows.

    Each patient/visit tensor is divided by its own maximum absolute activation,
    then the normalized absolute values are averaged over channels, visits, and
    patients.  This keeps sample amplitude from driving the montage.
    """

    value = np.asarray(activations)
    if value.ndim != 6 or any(int(size) <= 0 for size in value.shape):
        raise ValueError("activations must be [patient,visit,channel,Z,Y,X]")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("activations must be finite floating point")
    absolute = np.abs(value.astype(np.float64, copy=False))
    maxima = absolute.max(axis=(2, 3, 4, 5), keepdims=True)
    if np.any(maxima <= 0):
        raise ValueError("each patient/visit activation must contain a nonzero value")
    aggregate = (absolute / maxima).mean(axis=(0, 1, 2), dtype=np.float64)
    if not np.isfinite(aggregate).all() or np.any(aggregate < 0) or np.any(aggregate > 1):
        raise AssertionError("normalized activation aggregate escaped [0,1]")
    return np.ascontiguousarray(aggregate, dtype=np.float32)


def _activation_arrays(value: ActivationAggregate) -> dict[str, np.ndarray]:
    volume = np.asarray(value.activation_mean_zyx)
    if volume.dtype != np.float32 or volume.shape != ACTIVATION_SHAPE_ZYX:
        raise ValueError(
            f"activation aggregate must be float32 {ACTIVATION_SHAPE_ZYX}, "
            f"got {volume.dtype}/{volume.shape}"
        )
    if not np.isfinite(volume).all() or np.any(volume < 0) or np.any(volume > 1):
        raise ValueError("activation aggregate must be finite within [0,1]")
    if float(volume.max()) <= 0:
        raise ValueError("activation aggregate is empty")
    digest = str(value.checkpoint_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("activation aggregate checkpoint SHA-256 is malformed")
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "activation_mean_zyx": volume,
        "selected_patient_count": np.asarray(ACTIVATION_SELECTION_COUNT, dtype=np.int64),
        "visits_per_patient": np.asarray(ACTIVATION_VISIT_COUNT, dtype=np.int64),
        "source_row_count": np.asarray(
            ACTIVATION_SELECTION_COUNT * ACTIVATION_VISIT_COUNT, dtype=np.int64
        ),
        "channel_count": np.asarray(ACTIVATION_CHANNEL_COUNT, dtype=np.int64),
        "feature_shape_zyx": np.asarray(ACTIVATION_SHAPE_ZYX, dtype=np.int64),
        "normalization": np.asarray(ACTIVATION_NORMALIZATION),
        "selection_rule": np.asarray(ACTIVATION_SELECTION_RULE),
        "seed_base": np.asarray(2026, dtype=np.int64),
        "arm": np.asarray("N1"),
        "fold": np.asarray(0, dtype=np.int64),
        "stage": np.asarray("final"),
        "checkpoint_sha256": np.asarray(digest),
        "training_performed": np.asarray(False, dtype=bool),
        "outcomes_used": np.asarray(False, dtype=bool),
    }


def _private_activation_destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if not destination.name.endswith(".private.npz"):
        raise ValueError("activation aggregate must end in .private.npz")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite activation aggregate: {destination}")
    return destination


def write_private_activation_aggregate(
    value: ActivationAggregate, path: str | Path
) -> Path:
    """Atomically publish an owner-only aggregate with no IDs or source paths."""

    destination = _private_activation_destination(path)
    arrays = _activation_arrays(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    published = False
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.chmod(0o600)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        # A hard link is an atomic create-if-absent publication; it cannot
        # overwrite a file created after the preflight.
        os.link(temporary_path, destination)
        published = True
        temporary_path.unlink()
        destination.chmod(0o600)
        validate_activation_aggregate(destination, require_owner_only=True)
        return destination
    except Exception:
        if published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_activation_aggregate(
    path: str | Path,
    *,
    require_owner_only: bool = True,
    expected_checkpoint_sha256: str | None = None,
) -> ActivationAggregate:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or not source.name.endswith(".private.npz"):
        raise FileNotFoundError(f"private activation aggregate is absent: {source}")
    if require_owner_only and source.stat().st_mode & 0o077:
        raise PermissionError("private activation aggregate must be owner-only")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != ACTIVATION_AGGREGATE_KEYS:
            raise ValueError("activation aggregate key inventory drifted")
        volume = np.asarray(archive["activation_mean_zyx"])
        expected_scalars: Mapping[str, Any] = {
            "schema_version": 1,
            "selected_patient_count": ACTIVATION_SELECTION_COUNT,
            "visits_per_patient": ACTIVATION_VISIT_COUNT,
            "source_row_count": ACTIVATION_SELECTION_COUNT * ACTIVATION_VISIT_COUNT,
            "channel_count": ACTIVATION_CHANNEL_COUNT,
            "normalization": ACTIVATION_NORMALIZATION,
            "selection_rule": ACTIVATION_SELECTION_RULE,
            "seed_base": 2026,
            "arm": "N1",
            "fold": 0,
            "stage": "final",
            "training_performed": False,
            "outcomes_used": False,
        }
        for name, expected in expected_scalars.items():
            observed = _scalar(archive[name], name=name)
            if isinstance(expected, bool):
                matches = isinstance(observed, (bool, np.bool_)) and bool(observed) is expected
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(f"activation aggregate contract drifted at {name}")
        shape = np.asarray(archive["feature_shape_zyx"])
        if shape.dtype != np.int64 or tuple(shape.tolist()) != ACTIVATION_SHAPE_ZYX:
            raise ValueError("activation aggregate feature shape metadata drifted")
        digest = str(_scalar(archive["checkpoint_sha256"], name="checkpoint_sha256"))
    if expected_checkpoint_sha256 is not None:
        expected_digest = str(expected_checkpoint_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError("expected activation checkpoint SHA-256 is malformed")
        if digest != expected_digest:
            raise ValueError(
                "activation aggregate is not bound to the preregistered "
                "2026/N1/fold-0 checkpoint"
            )
    value = ActivationAggregate(
        activation_mean_zyx=np.ascontiguousarray(volume),
        checkpoint_sha256=digest,
    )
    _activation_arrays(value)
    return value


def _encode_normalized_abs_activation_maps(
    model: Any,
    image: Any,
    device: Any,
    *,
    visits: int = ACTIVATION_VISIT_COUNT,
    input_channels: int = 7,
    input_shape_zyx: tuple[int, int, int] = (112, 176, 160),
    channel_count: int = ACTIVATION_CHANNEL_COUNT,
    feature_shape_zyx: tuple[int, int, int] = ACTIVATION_SHAPE_ZYX,
) -> Any:
    """Call only ``model.encoder`` and return normalized channel-mean maps."""

    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError("activation image batch must be a torch Tensor")
    image = image.to(device, non_blocking=True)
    expected_tail = (int(visits), int(input_channels), *tuple(input_shape_zyx))
    if image.dtype != torch.float32 or tuple(image.shape[1:]) != expected_tail:
        raise ValueError("fixed N1 activation input shape/dtype drifted")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("fixed N1 activation input is non-finite")
    batch = int(image.shape[0])
    flat = image.reshape(batch * int(visits), *image.shape[2:])
    with torch.inference_mode():
        spatial = model.encoder(flat)
        expected_spatial = (
            batch * int(visits),
            int(channel_count),
            *tuple(feature_shape_zyx),
        )
        if (
            not isinstance(spatial, torch.Tensor)
            or spatial.dtype != torch.float32
            or tuple(spatial.shape) != expected_spatial
        ):
            raise ValueError("fixed N1 encoder activation shape/dtype drifted")
        absolute = spatial.abs()
        maxima = absolute.amax(dim=(1, 2, 3, 4), keepdim=True)
        if not bool(torch.isfinite(absolute).all()) or bool((maxima <= 0).any()):
            raise ValueError("fixed N1 encoder produced invalid activation")
        maps = (absolute / maxima).mean(dim=1)
    if (
        maps.dtype != torch.float32
        or not bool(torch.isfinite(maps).all())
        or bool((maps < 0).any())
        or bool((maps > 1).any())
    ):
        raise AssertionError("normalized encoder activation escaped finite [0,1]")
    return maps


def _aggregate_selected_encoder_loader(
    loader: Iterable[Mapping[str, Any]],
    selected_ids: Sequence[str],
    encode_batch: Callable[[Any], Any],
) -> np.ndarray:
    """Aggregate exactly 16 patients x 4 visits without retaining identifiers."""

    import torch

    expected_ids = tuple(str(value) for value in selected_ids)
    if (
        len(expected_ids) != ACTIVATION_SELECTION_COUNT
        or len(set(expected_ids)) != ACTIVATION_SELECTION_COUNT
        or any(not value for value in expected_ids)
    ):
        raise ValueError("formal activation aggregation requires exactly 16 unique IDs")
    accumulator = np.zeros(ACTIVATION_SHAPE_ZYX, dtype=np.float64)
    rows = 0
    observed_ids: list[str] = []
    for batch in loader:
        if "patient_id" not in batch or "image" not in batch:
            raise ValueError("activation batch lacks the patient/image contract")
        batch_ids = tuple(str(value) for value in batch["patient_id"])
        ordered = expected_ids[len(observed_ids) : len(observed_ids) + len(batch_ids)]
        if not batch_ids or batch_ids != ordered:
            raise RuntimeError("activation DataLoader changed deterministic patient order")
        maps = encode_batch(batch["image"])
        expected_shape = (
            len(batch_ids) * ACTIVATION_VISIT_COUNT,
            *ACTIVATION_SHAPE_ZYX,
        )
        if (
            not isinstance(maps, torch.Tensor)
            or maps.dtype != torch.float32
            or tuple(maps.shape) != expected_shape
            or not bool(torch.isfinite(maps).all())
            or bool((maps < 0).any())
            or bool((maps > 1).any())
        ):
            raise ValueError("normalized activation batch shape/value contract drifted")
        accumulator += maps.double().sum(dim=0).cpu().numpy()
        rows += int(maps.shape[0])
        observed_ids.extend(batch_ids)
    expected_rows = ACTIVATION_SELECTION_COUNT * ACTIVATION_VISIT_COUNT
    if tuple(observed_ids) != expected_ids or rows != expected_rows:
        raise RuntimeError("activation aggregation did not cover exact 16x4 rows")
    volume = np.ascontiguousarray(accumulator / rows, dtype=np.float32)
    if (
        not np.isfinite(volume).all()
        or np.any(volume < 0)
        or np.any(volume > 1)
        or float(volume.max()) <= 0
    ):
        raise ValueError("formal activation aggregate is empty or invalid")
    return volume


def build_formal_activation_aggregate(
    output_path: str | Path,
    *,
    device: str = "cuda:0",
    batch_size: int = 4,
    workers: int = 2,
) -> Path:
    """Build the one preregistered outcome-free N1/fold-0 activation aggregate.

    Only ``model.encoder`` is called.  No support/FTV target enters patient
    selection, model forward, or activation aggregation; the frozen Stage-B
    bundle is loaded for its locked cohort/cache/provenance contracts.  No probe
    is fitted and no checkpoint parameter is modified.
    """

    destination = _private_activation_destination(output_path)
    if int(batch_size) != 4 or int(workers) != 2:
        raise ValueError("formal activation aggregation requires batch_size=4/workers=2")

    import torch
    from torch.utils.data import DataLoader

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or torch_device.index is None or not torch.cuda.is_available():
        raise RuntimeError("formal activation aggregation requires explicit available CUDA")
    if torch_device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {torch_device}")

    from .runtime import load_selected_model, load_stage_b_bundle

    checkpoint, checkpoint_digest = _fixed_activation_checkpoint_binding()

    authorization, _, data = load_stage_b_bundle(verify_cache_files=False)
    from c1b_stage_b.data import StageBDataset, arm_cache, make_splits
    from c1b_stage_b.contracts import (
        LOGICAL_OBJECTIVE_CONTRACT,
        canonical_sha256,
        ordered_patient_sha256,
    )

    splits = make_splits(data.folds, 0, data.train_only_ids)
    formal_ids = tuple(splits.train_primary + splits.val + splits.test)
    if len(formal_ids) != 808 or len(set(formal_ids)) != 808:
        raise ValueError("activation population is not the exact 808 fold-assigned patients")
    selected_ids = select_patient_ids_by_sha256(formal_ids)
    dataset = StageBDataset(
        selected_ids,
        arm_cache("N1", data.legacy_cache, data.c1b_cache),
        transformed_ftv={},
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )
    model, checkpoint_payload = load_selected_model(checkpoint, torch_device)
    if (
        str(checkpoint_payload.get("arm", "")) != "N1"
        or int(checkpoint_payload.get("seed_base", -1)) != 2026
        or int(checkpoint_payload.get("fold", -1)) != 0
        or checkpoint_payload.get("selected") is not True
        or checkpoint_payload.get("test_data_used") is not False
    ):
        raise ValueError("fixed activation checkpoint identity drifted")
    if str(checkpoint_payload.get("stage_a_sentinel_sha256", "")) != authorization.sha256:
        raise ValueError("activation checkpoint and current Stage-A authorization differ")
    provenance = checkpoint_payload.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("activation checkpoint has no data provenance")
    if checkpoint_payload.get("data_provenance_sha256") != canonical_sha256(provenance):
        raise ValueError("activation checkpoint data provenance digest drifted")
    for name, value in data.provenance.items():
        if provenance.get(name) != value:
            raise ValueError(f"activation checkpoint data provenance differs at {name}")
    expected_provenance = {
        "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
        "validation_order_sha256": ordered_patient_sha256(splits.val),
        "test_patient_count_not_loaded": len(splits.test),
        "model_forward_fields": ["image"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
        "logical_objective_contract": dict(LOGICAL_OBJECTIVE_CONTRACT),
    }
    for name, value in expected_provenance.items():
        if provenance.get(name) != value:
            raise ValueError(f"activation checkpoint split/model contract differs at {name}")
    if checkpoint_payload.get("train_patient_sha256") != canonical_sha256(
        sorted(splits.train_all)
    ):
        raise ValueError("activation checkpoint training population drifted")
    if checkpoint_payload.get("val_patient_sha256") != canonical_sha256(
        sorted(splits.val)
    ):
        raise ValueError("activation checkpoint validation population drifted")
    selection_path = Path(str(checkpoint_payload.get("selection_path", ""))).resolve()
    if not selection_path.is_file() or checkpoint_payload.get("selection_sha256") != file_sha256(
        selection_path
    ):
        raise ValueError("activation checkpoint selection binding is absent or stale")

    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    volume = _aggregate_selected_encoder_loader(
        loader,
        selected_ids,
        lambda image: _encode_normalized_abs_activation_maps(
            model, image, torch_device
        ),
    )
    if tuple(parameter._version for parameter in model.parameters()) != parameter_versions:
        raise RuntimeError("checkpoint parameters mutated during activation aggregation")
    if file_sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("fixed activation checkpoint changed during aggregation")
    return write_private_activation_aggregate(
        ActivationAggregate(volume, checkpoint_digest), destination
    )


def load_figure_data(paths: FigureInputPaths) -> FigureData:
    """Read and validate every input before a public PNG can be created."""

    table1 = _load_csv(Path(paths.table1), TABLE1_COLUMNS, label="Table 1")
    table2 = _load_csv(Path(paths.table2), FTV_TABLE_COLUMNS, label="Table 2")
    table3 = _load_csv(Path(paths.table3), FTV_TABLE_COLUMNS, label="Table 3")
    table4 = _load_csv(Path(paths.table4), TABLE4_COLUMNS, label="Table 4")
    table5 = _load_csv(Path(paths.table5), TABLE5_COLUMNS, label="Table 5")
    table6 = _load_csv(Path(paths.table6), TABLE6_COLUMNS, label="Table 6")
    table7 = _load_csv(Path(paths.table7), TABLE7_COLUMNS, label="Table 7")
    _validate_table1(table1)
    table2_contract = _validate_ftv_table(
        table2, endpoints=(*TIMEPOINTS, "macro"), label="Table 2"
    )
    table3_contract = _validate_ftv_table(
        table3, endpoints=(*TRANSITIONS, "macro"), label="Table 3"
    )
    if table2_contract != table3_contract:
        raise ValueError(
            "Tables 2 and 3 disagree on conditional S3 or secondary-pooling presence"
        )
    table_stages, _ = table2_contract
    _validate_table4(table4, stages=set(table_stages))
    _validate_table5(table5)
    _validate_table6(table6)
    _validate_table7(table7)
    pooling_maps = load_pooling_weight_aggregates(paths.sidecar)
    _, expected_checkpoint_sha256 = _fixed_activation_checkpoint_binding()
    activation = validate_activation_aggregate(
        paths.activation_aggregate,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return FigureData(
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
        table5=table5,
        table6=table6,
        table7=table7,
        pooling_maps=pooling_maps,
        activation_mean_zyx=activation.activation_mean_zyx,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 13,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
         *, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=color,
        facecolor=f"{color}18",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#454545",
            transform=ax.transAxes,
        )
    )


def _figure01(data: FigureData) -> plt.Figure:
    frame = data.table1.set_index(["stage", "input_contract"])
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    ax.set_axis_off()
    _box(ax, (0.02, 0.64), 0.11, 0.16, "DCE7\n4 visits", color="#3769a6")
    _box(ax, (0.18, 0.64), 0.22, 0.16, "Frozen 3-D encoder\n4 residual blocks", color="#3769a6")
    _box(ax, (0.45, 0.64), 0.16, 0.16, "Spatial map F\n(before GAP)", color="#d65f4a")
    _box(
        ax,
        (0.66, 0.58),
        0.15,
        0.28,
        "FINAL poolings\nP0: GAP\nPVALID: source RF\n"
        "PLOCAL: 64-mm cube\nPORACLE: lesion RF",
        color="#2a9d8f",
    )
    _box(
        ax,
        (0.86, 0.64),
        0.12,
        0.16,
        "FINAL only\nFrozen Linear\n128→192 + LN",
        color="#7353ba",
    )
    for start, end in (
        ((0.13, 0.72), (0.18, 0.72)),
        ((0.40, 0.72), (0.45, 0.72)),
        ((0.61, 0.72), (0.66, 0.72)),
        ((0.81, 0.72), (0.86, 0.72)),
    ):
        _arrow(ax, start, end)
    ax.text(
        0.5,
        0.91,
        "Frozen-checkpoint spatial pooling audit (no training)",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.51,
        "FINAL PLOCAL+GLOBAL: concatenate projected local/global states (384-D); no fusion\n"
        "Conditional S3: raw pooled 64-D from features[2]; no frozen 64→192 projection",
        ha="center",
        transform=ax.transAxes,
        color="#4f4f4f",
        fontsize=8.5,
    )
    headers = ("Stage", "Input", "Input Z×Y×X", "Map C×Z×Y×X", "Jump", "RF")
    x_positions = (0.08, 0.21, 0.38, 0.60, 0.79, 0.91)
    for x, header in zip(x_positions, headers, strict=True):
        ax.text(x, 0.40, header, weight="bold", ha="center", transform=ax.transAxes)
    y_positions = (0.32, 0.24, 0.16, 0.08)
    keys = (("final", "legacy"), ("final", "c1b"), ("s3", "legacy"), ("s3", "c1b"))
    for y, key in zip(y_positions, keys, strict=True):
        row = frame.loc[key]
        values = (
            key[0].upper(),
            key[1].upper(),
            str(row["input_shape_zyx"]),
            f"{int(row['feature_channels'])}×{row['feature_shape_zyx']}",
            str(int(row["jump_input_voxels"])),
            f"{int(row['theoretical_receptive_field_input_voxels'])}³",
        )
        for x, value in zip(x_positions, values, strict=True):
            ax.text(x, y, value, ha="center", transform=ax.transAxes)
        ax.plot([0.03, 0.97], [y - 0.035, y - 0.035], color="#dddddd", lw=0.7,
                transform=ax.transAxes)
    return fig


def _projection_for_display(volume: np.ndarray) -> tuple[np.ndarray, float]:
    projection = np.asarray(volume, dtype=np.float64).max(axis=0)
    maximum = float(projection.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("pooling illustration received an empty aggregate map")
    return projection / maximum, maximum


def _figure02(data: FigureData) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), constrained_layout=True)
    images = []
    descriptions = {
        "P0": "uniform global",
        "PVALID": "mean valid-source RF",
        "PLOCAL": "fixed central 64 mm",
        "PORACLE": "non-deployable lesion RF",
    }
    for ax, pooling in zip(axes, ("P0", "PVALID", "PLOCAL", "PORACLE"), strict=True):
        display, raw_max = _projection_for_display(data.pooling_maps[pooling])
        image = ax.imshow(display, cmap="magma", origin="lower", vmin=0, vmax=1)
        images.append(image)
        ax.set_title(f"{pooling}\n{descriptions[pooling]}")
        ax.set_xlabel("X feature index")
        ax.set_ylabel("Y feature index")
        ax.text(
            0.02,
            0.02,
            f"raw max={raw_max:.3g}",
            color="white",
            fontsize=7,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
        )
    fig.colorbar(images[-1], ax=axes, shrink=0.78, label="per-map normalized weight")
    fig.suptitle(
        "Aggregate C1B pooling-support weights "
        "(no scalar FTV values; max projection over Z)"
    )
    return fig


def _heatmap_norm(values: np.ndarray, *, bounded: bool) -> tuple[Any, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("heatmap has no finite aggregate metrics")
    if bounded:
        return Normalize(vmin=-1.0, vmax=1.0), "coolwarm"
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    if math.isclose(low, high):
        high = low + 1.0
    if low < 0 < high:
        return TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high), "coolwarm"
    return Normalize(vmin=low, vmax=high), "viridis"


def _metric_heatmap(
    frame: pd.DataFrame, *, metric: str, title: str, bounded: bool
) -> plt.Figure:
    stages = [stage for stage in ("final", "s3") if stage in set(frame["stage"])]
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(8.2 * len(stages), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    primary = frame.loc[
        frame["endpoint"].astype(str).eq("macro")
        & ~frame["pooling"].astype(str).eq(_SECONDARY_POOLING)
    ]
    collected: list[float] = []
    matrices: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for stage in stages:
        poolings = _FINAL_POOLINGS if stage == "final" else _S3_POOLINGS
        matrix = np.full((len(ARMS), len(poolings)), np.nan, dtype=np.float64)
        for arm_index, arm in enumerate(ARMS):
            for pool_index, pooling in enumerate(poolings):
                rows = primary.loc[
                    primary["stage"].eq(stage)
                    & primary["arm"].eq(arm)
                    & primary["pooling"].eq(pooling)
                    & primary["availability"].eq("AVAILABLE")
                ]
                if rows.empty:
                    continue
                if len(rows) != len(SEEDS):
                    raise ValueError(f"{title} misses a seed at {stage}/{arm}/{pooling}")
                value = float(pd.to_numeric(rows[metric]).mean())
                if not math.isfinite(value):
                    raise ValueError(f"{title} has non-finite {stage}/{arm}/{pooling}")
                matrix[arm_index, pool_index] = value
                collected.append(value)
        matrices[stage] = (matrix, poolings)
    norm, cmap_name = _heatmap_norm(np.asarray(collected), bounded=bounded)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#ececec")
    last_image = None
    for axis_index, stage in enumerate(stages):
        ax = axes[0, axis_index]
        matrix, poolings = matrices[stage]
        last_image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(poolings)), [_POOLING_LABELS[value] for value in poolings],
                      rotation=28, ha="right")
        ax.set_yticks(range(len(ARMS)), ARMS)
        ax.set_title("Final map" if stage == "final" else "Conditional S3 map")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                label = "NA" if not np.isfinite(value) else f"{value:.2f}"
                color = "#555555" if not np.isfinite(value) else (
                    "white" if abs(float(norm(value)) - 0.5) > 0.27 else "black"
                )
                ax.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
    if last_image is None:
        raise AssertionError("no heatmap image was produced")
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82, label=metric)
    fig.suptitle(f"{title}\nmean across the two frozen seeds")
    return fig


def _figure03(data: FigureData) -> plt.Figure:
    return _metric_heatmap(
        data.table2,
        metric="spearman",
        title="Static FTV macro Spearman",
        bounded=True,
    )


def _figure04(data: FigureData) -> plt.Figure:
    return _metric_heatmap(
        data.table2,
        metric="natural_r2",
        title="Static FTV natural-scale macro R²",
        bounded=False,
    )


def _figure05(data: FigureData) -> plt.Figure:
    return _metric_heatmap(
        data.table3,
        metric="spearman",
        title="Literal ΔFTV macro Spearman",
        bounded=True,
    )


def _stage_arm_axes(stages: Sequence[str], *, width: float = 6.2) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(stages),
        2,
        figsize=(width * 2, 4.2 * len(stages)),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def _figure06(data: FigureData) -> plt.Figure:
    stages = [stage for stage in ("final", "s3") if stage in set(data.table4["stage"])]
    fig, axes = _stage_arm_axes(stages)
    for stage_index, stage in enumerate(stages):
        poolings = _FINAL_POOLINGS if stage == "final" else _S3_POOLINGS
        for arm_index, arm in enumerate(("N1", "N3")):
            ax = axes[stage_index, arm_index]
            group = data.table4.loc[
                data.table4["stage"].eq(stage) & data.table4["new_arm"].eq(arm)
            ]
            means: list[float] = []
            for pooling in poolings:
                rows = group.loc[group["pooling"].eq(pooling)].sort_values("seed_base")
                ratios = pd.to_numeric(rows["recovery_ratio"], errors="coerce").to_numpy(float)
                finite = ratios[np.isfinite(ratios)]
                means.append(float(finite.mean()) if finite.size else math.nan)
            x = np.arange(len(poolings), dtype=float)
            for index, (pooling, mean) in enumerate(zip(poolings, means, strict=True)):
                if math.isfinite(mean):
                    ax.bar(index, mean, width=0.65, color=_POOLING_COLORS[pooling], alpha=0.72)
                else:
                    ax.text(index, 0.02, "NA deficit", rotation=90, ha="center", va="bottom",
                            color="#777777", fontsize=7)
                rows = group.loc[group["pooling"].eq(pooling)].sort_values("seed_base")
                ratios = pd.to_numeric(rows["recovery_ratio"], errors="coerce").to_numpy(float)
                jitter = np.asarray((-0.09, 0.09))[: len(ratios)]
                finite_mask = np.isfinite(ratios)
                ax.scatter(
                    index + jitter[finite_mask],
                    ratios[finite_mask],
                    color="#202020",
                    s=22,
                    zorder=3,
                    marker="o",
                )
            ax.axhline(0, color="#555555", lw=0.8)
            ax.axhline(0.33, color="#457b9d", lw=0.9, ls="--", label="local gate 0.33")
            ax.axhline(0.50, color="#e76f51", lw=0.9, ls=":", label="oracle gate 0.50")
            ax.set_xticks(x, [_POOLING_LABELS[value] for value in poolings], rotation=28, ha="right")
            ax.set_ylabel("Legacy deficit recovery ratio")
            ax.set_title(f"{stage.upper()} · {arm} vs {'L1' if arm == 'N1' else 'L3'}")
            ax.grid(axis="y", color="#e4e4e4", linewidth=0.7)
            if stage_index == 0 and arm_index == 1:
                ax.legend(loc="best")
    fig.suptitle("Recovery of the matched legacy P0 Spearman deficit")
    return fig


def _figure07(data: FigureData) -> plt.Figure:
    stages = [stage for stage in ("final", "s3") if stage in set(data.table4["stage"])]
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(7.2 * len(stages), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    marker_by_pooling = {"PLOCAL": "o", "PLOCAL+GLOBAL": "s", "PORACLE": "D"}
    for stage_index, stage in enumerate(stages):
        ax = axes[0, stage_index]
        methods = ("PLOCAL", "PLOCAL+GLOBAL", "PORACLE") if stage == "final" else (
            "PLOCAL",
            "PORACLE",
        )
        for arm in ("N1", "N3"):
            for pooling in methods:
                rows = data.table4.loc[
                    data.table4["stage"].eq(stage)
                    & data.table4["new_arm"].eq(arm)
                    & data.table4["pooling"].eq(pooling)
                ].sort_values("seed_base")
                x = pd.to_numeric(rows["absolute_gain_vs_new_p0"], errors="coerce").to_numpy(float)
                y = pd.to_numeric(rows["recovery_ratio"], errors="coerce").to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y)
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=62,
                    marker=marker_by_pooling[pooling],
                    facecolor=_ARM_COLORS[arm],
                    edgecolor="#252525",
                    linewidth=0.6,
                    alpha=0.88,
                    label=f"{arm} · {_POOLING_LABELS[pooling]}",
                )
                for seed, x_value, y_value in zip(
                    rows["seed_base"].astype(int), x, y, strict=True
                ):
                    if np.isfinite(x_value) and np.isfinite(y_value):
                        ax.annotate(str(seed), (x_value, y_value), xytext=(3, 3),
                                    textcoords="offset points", fontsize=6)
        ax.axvline(0.10, color="#457b9d", ls="--", lw=0.9, label="local gain gate")
        ax.axvline(0.15, color="#e76f51", ls=":", lw=0.9, label="oracle gain gate")
        ax.axhline(0.33, color="#457b9d", ls="--", lw=0.8)
        ax.axhline(0.50, color="#e76f51", ls=":", lw=0.8)
        ax.axvline(0, color="#555555", lw=0.7)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_xlabel("Absolute macro Spearman gain vs new-arm P0")
        ax.set_ylabel("Matched legacy deficit recovery")
        ax.set_title("Final spatial map" if stage == "final" else "Conditional S3 map")
        ax.grid(color="#e8e8e8", linewidth=0.6)
        ax.legend(loc="best", ncol=1, fontsize=7)
    fig.suptitle("Oracle versus outcome-free local recovery")
    return fig


def _figure08(data: FigureData) -> plt.Figure:
    targets = ("padding_fraction", "valid_source_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(_NUISANCE_POOLINGS), dtype=float)
    width = 0.34
    for target_index, target in enumerate(targets):
        ax = axes[target_index]
        for arm_index, arm in enumerate(("N1", "N3")):
            offset = (arm_index - 0.5) * width
            means: list[float] = []
            seed_values: list[np.ndarray] = []
            for pooling in _NUISANCE_POOLINGS:
                rows = data.table5.loc[
                    data.table5["arm"].eq(arm)
                    & data.table5["pooling"].eq(pooling)
                    & data.table5["target_name"].eq(target)
                    & data.table5["endpoint"].eq("macro")
                    & data.table5["availability"].eq("AVAILABLE")
                ].sort_values("seed_base")
                values = pd.to_numeric(rows["natural_r2"]).to_numpy(float)
                if len(values) != 2 or not np.isfinite(values).all():
                    raise ValueError(f"Figure 08 misses {arm}/{pooling}/{target}")
                means.append(float(values.mean()))
                seed_values.append(values)
            positions = x + offset
            ax.bar(
                positions,
                means,
                width=width,
                color=_ARM_COLORS[arm],
                alpha=0.72,
                label=arm,
            )
            for position, values in zip(positions, seed_values, strict=True):
                ax.scatter(position + np.asarray((-0.045, 0.045)), values, color="#202020",
                           s=18, zorder=3)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.axhline(0.20, color="#e76f51", ls="--", lw=0.9, label="P0 R² gate 0.20")
        ax.set_xticks(x, _NUISANCE_POOLINGS)
        ax.set_ylabel("Macro natural R²")
        ax.set_title(target.replace("_", " "))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
        ax.legend(loc="best")
    fig.suptitle("Padding and valid-source nuisance decodability")
    return fig


def _figure09(data: FigureData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    for arm in ("N1", "N3"):
        for pooling in _NUISANCE_POOLINGS:
            for seed in SEEDS:
                ftv = data.table2.loc[
                    data.table2["stage"].eq("final")
                    & data.table2["seed_base"].eq(seed)
                    & data.table2["arm"].eq(arm)
                    & data.table2["pooling"].eq(pooling)
                    & data.table2["endpoint"].eq("macro")
                    & data.table2["availability"].eq("AVAILABLE"),
                    "spearman",
                ]
                nuisance = data.table5.loc[
                    data.table5["seed_base"].eq(seed)
                    & data.table5["arm"].eq(arm)
                    & data.table5["pooling"].eq(pooling)
                    & data.table5["endpoint"].eq("macro")
                    & data.table5["availability"].eq("AVAILABLE"),
                    "spearman",
                ]
                if len(ftv) != 1 or len(nuisance) != len(NUISANCE_TARGETS):
                    raise ValueError(f"Figure 09 comparison matrix missing {seed}/{arm}/{pooling}")
                x_value = float(pd.to_numeric(nuisance).mean())
                y_value = float(pd.to_numeric(ftv).iloc[0])
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    raise ValueError("Figure 09 encountered a non-finite comparison")
                ax.scatter(
                    x_value,
                    y_value,
                    s=65,
                    marker={"P0": "o", "PVALID": "s", "PLOCAL": "D"}[pooling],
                    facecolor=_ARM_COLORS[arm],
                    edgecolor="#222222",
                    linewidth=0.6,
                )
                ax.annotate(
                    f"{arm}/{pooling}/{seed}",
                    (x_value, y_value),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                )
    ax.axhline(0, color="#666666", lw=0.8)
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Mean nuisance macro Spearman (10 outcome-free targets)")
    ax.set_ylabel("Static FTV macro Spearman")
    ax.grid(color="#e8e8e8", linewidth=0.7)
    ax.set_title("FTV information versus acquisition/geometry information")
    return fig


def _figure10(data: FigureData) -> plt.Figure:
    frame = data.table6.loc[data.table6["analysis"].eq("occupancy_quartile")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    quartiles = ("Q1", "Q2", "Q3", "Q4")
    endpoint_colors = dict(zip(TIMEPOINTS, ("#264653", "#2a9d8f", "#e9c46a", "#e76f51"), strict=True))
    for seed_index, seed in enumerate(SEEDS):
        for endpoint in TIMEPOINTS:
            rows = frame.loc[
                frame["seed_base"].eq(seed) & frame["endpoint"].eq(endpoint)
            ].set_index("stratum").loc[list(quartiles)]
            style = "-" if seed_index == 0 else "--"
            label = f"{endpoint} · seed {seed}"
            axes[0].plot(
                quartiles,
                pd.to_numeric(rows["n1_minus_l1_spearman"]),
                color=endpoint_colors[endpoint],
                linestyle=style,
                marker="o",
                markersize=3.8,
                label=label,
            )
            axes[1].plot(
                quartiles,
                pd.to_numeric(rows["n1_minus_l1_mae"]),
                color=endpoint_colors[endpoint],
                linestyle=style,
                marker="o",
                markersize=3.8,
                label=label,
            )
    for ax in axes:
        ax.axhline(0, color="#555555", lw=0.8)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_xlabel("Lesion occupancy quartile (pooled qcut)")
    axes[0].set_ylabel("N1 − L1 Spearman")
    axes[0].set_title("Correlation degradation")
    axes[1].set_ylabel("N1 − L1 MAE")
    axes[1].set_title("Absolute-error degradation")
    axes[1].legend(loc="best", ncol=2, fontsize=6.8)
    fig.suptitle("Occupancy-stratified N1 versus L1 P0 degradation")
    return fig


def _figure11(data: FigureData) -> plt.Figure:
    frame = data.table7.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    ax = axes[0]
    for arm_index, arm in enumerate(ARMS):
        group = frame.loc[frame["arm"].eq(arm)].sort_values(["seed", "fold"])
        offsets = np.linspace(-0.28, 0.28, len(group))
        x_values = arm_index + offsets
        selected = pd.to_numeric(group["selected_epoch"]).to_numpy(float)
        observed = pd.to_numeric(group["observed_max_epoch"]).to_numpy(float)
        for x_value, selected_value, observed_value in zip(
            x_values, selected, observed, strict=True
        ):
            ax.plot([x_value, x_value], [selected_value, observed_value], color="#b7b7b7", lw=0.8)
        ax.scatter(x_values, selected, color=_ARM_COLORS[arm], s=25, marker="o",
                   label=f"{arm} selected")
        ax.scatter(x_values, observed, facecolor="white", edgecolor=_ARM_COLORS[arm],
                   s=28, marker="o", linewidth=1.0)
    ax.axhline(12, color="#e76f51", ls="--", lw=1.0, label="configured epoch 12")
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.set_ylabel("Epoch")
    ax.set_title("Selected epoch (filled) and observed maximum (open)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.legend(loc="best", ncol=2, fontsize=6.8)

    ax = axes[1]
    for arm_index, arm in enumerate(ARMS):
        values = pd.to_numeric(
            frame.loc[frame["arm"].eq(arm), "last_three_normalized_validation_state_slope"]
        ).to_numpy(float)
        offsets = np.linspace(-0.22, 0.22, len(values))
        ax.scatter(
            arm_index + offsets,
            values,
            color=_ARM_COLORS[arm],
            edgecolor="#222222",
            linewidth=0.35,
            s=28,
            alpha=0.82,
        )
        median = float(np.median(values))
        ax.plot([arm_index - 0.28, arm_index + 0.28], [median, median], color="#111111", lw=2)
    ax.axhline(-0.005, color="#e76f51", ls="--", lw=1.0, label="undertraining slope gate")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.set_ylabel("Normalized last-three validation-loss slope")
    ax.set_title("Frozen history tail (line = arm median)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.legend(loc="best")
    fig.suptitle("Selected checkpoint location and training-budget audit")
    return fig


def _orient(image: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(image), k=1)


def _figure12(data: FigureData) -> plt.Figure:
    volume = np.asarray(data.activation_mean_zyx, dtype=np.float64)
    if volume.shape != ACTIVATION_SHAPE_ZYX:
        raise ValueError("activation montage volume shape drifted")
    z, y, x = (size // 2 for size in volume.shape)
    panels = (
        ("Axial central slice", _orient(volume[z, :, :])),
        ("Coronal central slice", _orient(volume[:, y, :])),
        ("Sagittal central slice", _orient(volume[:, :, x])),
        ("Axial max projection", _orient(volume.max(axis=0))),
        ("Coronal max projection", _orient(volume.max(axis=1))),
        ("Sagittal max projection", _orient(volume.max(axis=2))),
    )
    vmax = float(volume.max())
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.7), constrained_layout=True)
    image = None
    for ax, (title, panel) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(panel, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is None:
        raise AssertionError("activation montage has no panels")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82,
                 label="mean normalized |encoder activation|")
    fig.suptitle(
        "Outcome-free aggregate activation montage\n"
        "fixed final N1 map · 16 hash-selected patients × 4 visits"
    )
    return fig


_FIGURE_BUILDERS: tuple[Callable[[FigureData], plt.Figure], ...] = (
    _figure01,
    _figure02,
    _figure03,
    _figure04,
    _figure05,
    _figure06,
    _figure07,
    _figure08,
    _figure09,
    _figure10,
    _figure11,
    _figure12,
)


def _save_figure_temporary(
    figure: plt.Figure, *, destination: Path, title: str
) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=FIGURE_DPI,
            facecolor="white",
            metadata={"Software": "c1b_spatial_audit", "Title": title},
        )
        temporary.chmod(0o644)
        with temporary.open("rb") as stream:
            signature = stream.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                raise ValueError("matplotlib did not create a PNG")
            os.fsync(stream.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def render_public_figures(
    paths: FigureInputPaths, *, output_dir: str | Path
) -> dict[str, Path]:
    """Create exactly twelve 200-dpi public PNGs with atomic non-overwrite."""

    if len(FIGURE_FILENAMES) != 12 or len(_FIGURE_BUILDERS) != 12:
        raise AssertionError("public figure inventory must contain exactly twelve figures")
    if len(set(FIGURE_FILENAMES)) != 12:
        raise AssertionError("public figure filenames must be unique")

    # Validate every public/private aggregate first.  A malformed or incomplete
    # matrix therefore cannot leave a partially published figure set.
    data = load_figure_data(paths)
    destination_root = Path(output_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root.chmod(0o755)
    existing_pngs = sorted(destination_root.glob("*.png"))
    if existing_pngs:
        raise FileExistsError(
            f"refusing to overwrite or mix public figure PNGs: {existing_pngs[0]}"
        )
    outputs = {
        name: destination_root / name
        for name in FIGURE_FILENAMES
    }
    temporaries: list[tuple[Path, Path]] = []
    published: list[Path] = []
    _style()
    try:
        for filename, builder in zip(FIGURE_FILENAMES, _FIGURE_BUILDERS, strict=True):
            figure: plt.Figure | None = None
            try:
                figure = builder(data)
                temporary = _save_figure_temporary(
                    figure,
                    destination=outputs[filename],
                    title=filename.removesuffix(".png"),
                )
                temporaries.append((temporary, outputs[filename]))
            finally:
                if figure is not None:
                    plt.close(figure)
        for temporary, destination in temporaries:
            os.link(temporary, destination)
            temporary.unlink()
            destination.chmod(0o644)
            published.append(destination)
        if tuple(sorted(path.name for path in destination_root.glob("*.png"))) != tuple(
            sorted(FIGURE_FILENAMES)
        ):
            raise RuntimeError("published public PNG inventory drifted")
        return outputs
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)
        plt.close("all")


__all__ = [
    "ACTIVATION_AGGREGATE_KEYS",
    "ACTIVATION_NORMALIZATION",
    "ACTIVATION_SELECTION_RULE",
    "ActivationAggregate",
    "FIGURE_DPI",
    "FIGURE_FILENAMES",
    "FigureData",
    "FigureInputPaths",
    "TABLE7_COLUMNS",
    "aggregate_normalized_abs_activations",
    "build_formal_activation_aggregate",
    "load_figure_data",
    "load_pooling_weight_aggregates",
    "render_public_figures",
    "select_patient_ids_by_sha256",
    "validate_activation_aggregate",
    "write_private_activation_aggregate",
]
