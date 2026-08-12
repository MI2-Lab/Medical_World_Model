"""Core contracts for the frozen non-FTV phenotype decodability audit.

The module deliberately separates three stages:

1. authenticate and align immutable patient-level inputs;
2. fit every target transform, residualizer, feature scaler, and ridge path on
   the outer-training partition only, using validation only for alpha choice;
3. aggregate untouched outer-test predictions without reading pCR outcomes.

Identifier-bearing arrays and predictions are owner-private.  Public outputs
contain only aggregate metrics, counts, hashes, and model-selection metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.linear_model import Ridge


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_ROOT.parents[1]
VISITS = ("T0", "T1", "T2", "T3")
INTERVALS = ("T0->T1", "T1->T2", "T2->T3")
FAMILIES = ("FTV", "LD", "SPH", "BPE")
PRIMARY_RESIDUAL_TARGETS = ("LD", "SPH", "BPE")
SECONDARY_RESIDUAL_TARGETS = ("SPH", "BPE")
MAIN_REPRESENTATIONS = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7")
ORACLE_TO_MATCHED = {
    "Z5": "Z4_MATCHED_Z5",
    "Z6": "Z4_MATCHED_Z6",
    "Z7": "Z4_MATCHED_Z7",
}
REGION_INDEX = {"Z5": 0, "Z6": 1, "Z7": 2}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported audit config schema")
    if config.get("experiment") != "nonftv_phenotype_decodability_audit":
        raise ValueError("wrong experiment config")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def authenticate(path: str | Path, expected_sha256: str, label: str) -> str:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    observed = file_sha256(resolved)
    if observed != str(expected_sha256):
        raise ValueError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {observed}"
        )
    return observed


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o644)
        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: str | Path, payload: Mapping[str, Any], *, private: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class TargetDataset:
    patient_ids: np.ndarray
    trial_ids: np.ndarray
    values: Mapping[str, np.ndarray]
    patient_to_index: Mapping[str, int]
    patient_set_sha256: str
    workbook_max_abs_difference: Mapping[str, float]


def _assign_endpoint(
    destination: np.ndarray,
    row_index: int,
    visit_index: int,
    value: float,
    *,
    label: str,
) -> None:
    previous = destination[row_index, visit_index]
    if np.isfinite(previous) and not np.isclose(previous, value, rtol=0.0, atol=1e-12):
        raise ValueError(f"inconsistent repeated endpoint for {label}")
    destination[row_index, visit_index] = float(value)


def load_targets(config: Mapping[str, Any]) -> TargetDataset:
    paths = config["paths"]
    target_path = resolve_path(paths["eligible_target_table"])
    authenticate(
        target_path,
        paths["eligible_target_table_sha256"],
        "eligible target table",
    )
    required_columns = {
        "patient_id",
        "trial_id",
        "transition",
        "start_visit",
        "end_visit",
        *{
            f"{prefix}_{suffix}"
            for prefix in ("ftv", "ld", "sphericity", "bpe")
            for suffix in ("start", "end", "valid")
        },
    }
    raw = pd.read_csv(target_path, usecols=lambda name: name in required_columns)
    if set(raw.columns) != required_columns:
        raise ValueError(f"target table schema drift: {sorted(set(raw.columns) ^ required_columns)}")
    if len(raw) != int(config["frozen"]["patient_count"]) * len(INTERVALS):
        raise ValueError("target table does not contain 375 x 3 rows")
    if not all(raw[f"{prefix}_valid"].astype(bool).all() for prefix in ("ftv", "ld", "sphericity", "bpe")):
        raise ValueError("frozen target table unexpectedly contains invalid endpoints")

    patient_ids = np.asarray(sorted(raw["patient_id"].astype(str).unique()), dtype=str)
    if len(patient_ids) != int(config["frozen"]["patient_count"]):
        raise ValueError("eligible target patient count drifted")
    patient_to_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    trial_by_patient = raw.groupby("patient_id", sort=False)["trial_id"].nunique()
    if not (trial_by_patient == 1).all():
        raise ValueError("patient-to-trial mapping is not one-to-one")
    trial_lookup = raw.drop_duplicates("patient_id").set_index("patient_id")["trial_id"]
    trial_ids = np.asarray([int(trial_lookup.loc[patient_id]) for patient_id in patient_ids], dtype=np.int64)

    values = {family: np.full((len(patient_ids), len(VISITS)), np.nan, dtype=np.float64) for family in FAMILIES}
    prefix_by_family = {"FTV": "ftv", "LD": "ld", "SPH": "sphericity", "BPE": "bpe"}
    for row in raw.itertuples(index=False):
        patient_id = str(row.patient_id)
        row_index = patient_to_index[patient_id]
        try:
            start_index = VISITS.index(str(row.start_visit))
            end_index = VISITS.index(str(row.end_visit))
        except ValueError as error:
            raise ValueError("unknown visit in target table") from error
        if end_index != start_index + 1:
            raise ValueError("target table contains a non-adjacent interval")
        for family, prefix in prefix_by_family.items():
            _assign_endpoint(
                values[family],
                row_index,
                start_index,
                float(getattr(row, f"{prefix}_start")),
                label=f"{patient_id}/{family}/{VISITS[start_index]}",
            )
            _assign_endpoint(
                values[family],
                row_index,
                end_index,
                float(getattr(row, f"{prefix}_end")),
                label=f"{patient_id}/{family}/{VISITS[end_index]}",
            )
    for family, array in values.items():
        if array.shape != (len(patient_ids), 4) or not np.isfinite(array).all():
            raise ValueError(f"incomplete reconstructed target matrix: {family}")

    workbook_path = resolve_path(paths["workbook"])
    authenticate(workbook_path, paths["workbook_sha256"], "Goal 6 workbook")
    workbook_columns = ["CLINICAL-TRIAL-SUBJECT-ID"] + [
        field for family in FAMILIES for field in config["targets"][family]
    ]
    workbook = pd.read_excel(
        workbook_path,
        sheet_name=str(paths["workbook_sheet"]),
        usecols=workbook_columns,
    )
    workbook["CLINICAL-TRIAL-SUBJECT-ID"] = pd.to_numeric(
        workbook["CLINICAL-TRIAL-SUBJECT-ID"], errors="raise"
    ).astype(np.int64)
    if workbook["CLINICAL-TRIAL-SUBJECT-ID"].duplicated().any():
        raise ValueError("Goal 6 workbook contains duplicate trial IDs")
    workbook = workbook.set_index("CLINICAL-TRIAL-SUBJECT-ID")
    differences: dict[str, float] = {}
    for family in FAMILIES:
        expected = workbook.loc[trial_ids, config["targets"][family]].to_numpy(dtype=np.float64)
        observed = np.asarray(values[family], dtype=np.float64)
        maximum = float(np.max(np.abs(expected - observed)))
        if not np.isfinite(maximum) or maximum > 1e-10:
            raise ValueError(f"{family} target table differs from the real workbook: {maximum}")
        differences[family] = maximum

    return TargetDataset(
        patient_ids=patient_ids,
        trial_ids=trial_ids,
        values=values,
        patient_to_index=patient_to_index,
        patient_set_sha256=ordered_sha256(patient_ids),
        workbook_max_abs_difference=differences,
    )


def load_fold_splits(config: Mapping[str, Any], targets: TargetDataset) -> Mapping[int, np.ndarray]:
    paths = config["paths"]
    fold_path = resolve_path(paths["fold_manifest"])
    authenticate(fold_path, paths["fold_manifest_sha256"], "fold manifest")
    # Intentionally do not parse label_pcr: pCR is forbidden for this audit.
    folds = pd.read_csv(fold_path, usecols=["patient_id", "fold", "split"])
    folds["patient_id"] = folds["patient_id"].astype(str)
    allowed = set(targets.patient_ids.tolist())
    folds = folds.loc[folds["patient_id"].isin(allowed)].copy()
    output: dict[int, np.ndarray] = {}
    for fold in config["frozen"]["folds"]:
        current = folds.loc[folds["fold"] == int(fold)]
        if len(current) != len(targets.patient_ids) or current["patient_id"].nunique() != len(targets.patient_ids):
            raise ValueError(f"fold {fold} does not cover the exact target cohort")
        lookup = current.set_index("patient_id")["split"]
        labels = np.asarray([str(lookup.loc[patient_id]) for patient_id in targets.patient_ids], dtype=str)
        if set(labels.tolist()) != {"train", "val", "test"}:
            raise ValueError(f"fold {fold} split labels drifted")
        output[int(fold)] = labels
    test_counts = np.zeros(len(targets.patient_ids), dtype=np.int64)
    for labels in output.values():
        test_counts += labels == "test"
    if not np.array_equal(test_counts, np.ones_like(test_counts)):
        raise ValueError("each target patient must be outer-test exactly once")
    return output


@dataclass(frozen=True)
class FeatureCell:
    seed: int
    arm: str
    fold: int
    patient_ids: np.ndarray
    splits: np.ndarray
    representations: Mapping[str, np.ndarray]
    validity: Mapping[str, np.ndarray]
    selected_epoch: int
    provenance: Mapping[str, Any]


def _project_z1(
    response: np.ndarray,
    spatial_mean: np.ndarray,
    checkpoint_path: Path,
    *,
    parity_rtol: float,
    parity_atol: float,
    parity_max_abs: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("selected checkpoint is not a valid mapping")
    state_dict = checkpoint["state_dict"]
    projector = nn.Sequential(
        nn.Linear(192, 384),
        nn.LayerNorm(384),
        nn.GELU(),
        nn.Linear(384, 192),
    )
    prefix = "projector.network."
    projector_state = {
        key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
    }
    projector.load_state_dict(projector_state, strict=True)
    response_projection = nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192))
    response_prefix = "response_projection."
    response_projection.load_state_dict(
        {
            key[len(response_prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(response_prefix)
        },
        strict=True,
    )
    projector.eval()
    response_projection.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(response, dtype=np.float32))
        projected = projector(tensor).to(dtype=torch.float32, device="cpu").numpy()
        recomputed_response = response_projection(
            torch.from_numpy(np.asarray(spatial_mean, dtype=np.float32))
        ).to(dtype=torch.float32, device="cpu").numpy()
    if projected.shape != response.shape or not np.isfinite(projected).all():
        raise ValueError("derived Z1 tensor is invalid")
    projection_parity_max_abs = float(
        np.max(np.abs(recomputed_response - np.asarray(response, dtype=np.float32)))
    )
    if projection_parity_max_abs > float(parity_max_abs) or not np.allclose(
        recomputed_response,
        np.asarray(response, dtype=np.float32),
        rtol=float(parity_rtol),
        atol=float(parity_atol),
    ):
        raise ValueError(
            "recomputed response_projection(Z3) does not match frozen Z2: "
            f"max_abs={projection_parity_max_abs}"
        )
    evidence = {
        "selected": bool(checkpoint.get("selected")),
        "test_data_used": bool(checkpoint.get("test_data_used")),
        "pcr_used": bool(checkpoint.get("pcr_used")),
        "delta_ftv_used": bool(checkpoint.get("delta_ftv_used")),
        "selected_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_seed_base": int(checkpoint.get("seed_base", -1)),
        "checkpoint_fold": int(checkpoint.get("fold", -1)),
        "checkpoint_arm": str(checkpoint.get("arm", "")),
        "architecture": str(checkpoint.get("architecture", "")),
        "projector": "Linear(192,384)-LayerNorm-GELU-Linear(384,192)",
        "z3_to_z2_projection_parity_max_abs": projection_parity_max_abs,
        "z3_to_z2_projection_parity_rtol": float(parity_rtol),
        "z3_to_z2_projection_parity_atol": float(parity_atol),
        "z3_to_z2_projection_parity_max_abs_limit": float(parity_max_abs),
    }
    if evidence["selected"] is not True or evidence["test_data_used"] is not False:
        raise ValueError("checkpoint is not a selected, test-blind checkpoint")
    if evidence["pcr_used"] is not False:
        raise ValueError("checkpoint reports pCR use")
    return projected, evidence


def load_feature_cell(
    config: Mapping[str, Any],
    targets: TargetDataset,
    fold_splits: Mapping[int, np.ndarray],
    *,
    seed: int,
    arm: str,
    fold: int,
) -> FeatureCell:
    local_root = resolve_path(config["paths"]["local_feature_root"])
    checkpoint_root = resolve_path(config["paths"]["checkpoint_root"])
    spatial_root = resolve_path(config["paths"]["spatial_feature_root"])
    relative = Path(f"seed_{seed}") / arm / f"fold_{fold}"
    local_path = local_root / relative / "response_state.private.npz"
    local_metadata_path = local_path.with_suffix(".metadata.json")
    checkpoint_path = checkpoint_root / relative / str(config["frozen"]["checkpoint_filename"])
    spatial_path = spatial_root / relative / "spatial_statistics.private.npz"
    spatial_metadata_path = spatial_path.with_suffix(".metadata.json")
    for label, path in (
        ("Z2 feature", local_path),
        ("Z2 metadata", local_metadata_path),
        ("selected checkpoint", checkpoint_path),
        ("spatial feature", spatial_path),
        ("spatial metadata", spatial_metadata_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    local_metadata = load_json(local_metadata_path)
    spatial_metadata = load_json(spatial_metadata_path)
    completion = load_json(
        spatial_root / "feature_matrix_complete.private.json"
    )
    local_sha = file_sha256(local_path)
    spatial_sha = file_sha256(spatial_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    if local_sha != local_metadata.get("feature_sha256"):
        raise ValueError("Z2 feature hash differs from metadata")
    if spatial_sha != spatial_metadata.get("feature_sha256"):
        raise ValueError("spatial feature hash differs from metadata")
    completion_matches = [
        row
        for row in completion.get("cells", [])
        if int(row.get("seed", -1)) == int(seed)
        and str(row.get("arm", "")) == str(arm)
        and int(row.get("fold", -1)) == int(fold)
    ]
    if len(completion_matches) != 1 or completion_matches[0].get("feature_sha256") != spatial_sha:
        raise ValueError("spatial feature hash differs from the immutable completion marker")
    if checkpoint_sha != local_metadata.get("checkpoint_sha256") or checkpoint_sha != spatial_metadata.get("checkpoint_sha256"):
        raise ValueError("checkpoint hash differs across frozen feature assets")
    if spatial_metadata.get("reference_feature_sha256") != local_sha:
        raise ValueError("spatial asset is not bound to the current Z2 feature")
    selection_path = checkpoint_path.with_name("selection.json")
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing selection record: {selection_path}")
    selection_sha = file_sha256(selection_path)
    if (
        selection_sha != local_metadata.get("selection_sha256")
        or selection_sha != spatial_metadata.get("selection_sha256")
    ):
        raise ValueError("selection hash differs across frozen assets")
    parity = spatial_metadata.get("p1_projection_parity", {})
    if parity.get("allclose") is not True or float(parity.get("max_abs_difference", math.inf)) > 1e-6:
        raise ValueError("spatial P1/Z2 parity contract failed")
    identities = {
        "seed_base": int(seed),
        "arm": str(arm),
        "fold": int(fold),
    }
    for metadata in (local_metadata, spatial_metadata):
        for key, expected in identities.items():
            observed = metadata.get(key)
            observed = int(observed) if key in {"seed_base", "fold"} else str(observed)
            if observed != expected:
                raise ValueError(f"feature identity mismatch at {key}")

    with np.load(local_path, allow_pickle=False) as archive:
        patient_ids = archive["patient_id"].astype(str)
        splits = archive["split"].astype(str)
        z2 = archive["response_state"].astype(np.float32, copy=True)
    with np.load(spatial_path, allow_pickle=False) as archive:
        spatial_ids = archive["patient_id"].astype(str)
        spatial_splits = archive["split"].astype(str)
        mean = archive["mean"].astype(np.float32, copy=True)
        std = archive["std"].astype(np.float32, copy=True)
        oracle_mean = archive["oracle_mean"].astype(np.float32, copy=True)
        oracle_std = archive["oracle_std"].astype(np.float32, copy=True)
        oracle_valid = archive["oracle_valid"].astype(bool, copy=True)
        oracle_regions = tuple(archive["oracle_regions"].astype(str).tolist())
    if not np.array_equal(patient_ids, spatial_ids) or not np.array_equal(splits, spatial_splits):
        raise ValueError("Z2 and spatial feature patient/split order differs")
    if len(np.unique(patient_ids)) != len(patient_ids):
        raise ValueError("feature cell contains duplicate patient identifiers")
    if ordered_sha256(patient_ids) != local_metadata.get("patient_order_sha256") or ordered_sha256(patient_ids) != spatial_metadata.get("patient_order_sha256"):
        raise ValueError("feature patient-order hash differs from metadata")
    if ordered_sha256(splits) != spatial_metadata.get("split_order_sha256"):
        raise ValueError("feature split-order hash differs from metadata")
    if len(patient_ids) != int(config["frozen"]["full_feature_patient_count"]):
        raise ValueError("feature cell patient count drifted")
    if z2.shape != (len(patient_ids), 4, 192) or mean.shape != (len(patient_ids), 4, 128):
        raise ValueError("feature cell tensor shape drifted")
    if std.shape != mean.shape or oracle_mean.shape != (len(patient_ids), 4, 4, 128):
        raise ValueError("spatial statistic tensor shape drifted")
    if oracle_std.shape != oracle_mean.shape or oracle_valid.shape != (len(patient_ids), 4, 4):
        raise ValueError("oracle tensor shape drifted")
    if oracle_regions != ("CORE", "PERI10", "PERI20", "LOCAL_REST"):
        raise ValueError("oracle region order drifted")
    if not all(np.isfinite(array).all() for array in (z2, mean, std, oracle_mean, oracle_std)):
        raise FloatingPointError("feature cell contains NaN or Inf")

    index = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    try:
        subset = np.asarray([index[patient_id] for patient_id in targets.patient_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("target cohort is not a subset of the feature cohort") from error
    expected_splits = fold_splits[int(fold)]
    if not np.array_equal(splits[subset], expected_splits):
        raise ValueError("feature split labels differ from the locked fold manifest")

    parity_config = config["projection_parity"]
    z1_full, checkpoint_evidence = _project_z1(
        z2,
        mean,
        checkpoint_path,
        parity_rtol=float(parity_config["direct_cpu_recompute_rtol"]),
        parity_atol=float(parity_config["direct_cpu_recompute_atol"]),
        parity_max_abs=float(parity_config["direct_cpu_recompute_max_abs_lte"]),
    )
    if (
        checkpoint_evidence["checkpoint_seed_base"] != int(seed)
        or checkpoint_evidence["checkpoint_fold"] != int(fold)
        or checkpoint_evidence["checkpoint_arm"] != str(arm)
        or checkpoint_evidence["selected_epoch"]
        != int(local_metadata.get("selected_epoch", -1))
    ):
        raise ValueError("selected checkpoint identity/epoch differs from feature metadata")
    z1 = z1_full[subset]
    z2 = z2[subset]
    mean = mean[subset]
    std = std[subset]
    oracle_mean = oracle_mean[subset]
    oracle_std = oracle_std[subset]
    oracle_valid = oracle_valid[subset]
    representations = {
        "Z1": z1,
        "Z2": z2,
        "Z3": mean,
        "Z4": np.concatenate((mean, std), axis=-1),
        "Z5": np.concatenate((oracle_mean[:, :, 0], oracle_std[:, :, 0]), axis=-1),
        "Z6": np.concatenate((oracle_mean[:, :, 1], oracle_std[:, :, 1]), axis=-1),
        "Z7": np.concatenate((oracle_mean[:, :, 2], oracle_std[:, :, 2]), axis=-1),
    }
    full_valid = np.ones((len(targets.patient_ids), 4), dtype=bool)
    validity = {
        "Z1": full_valid.copy(),
        "Z2": full_valid.copy(),
        "Z3": full_valid.copy(),
        "Z4": full_valid.copy(),
        "Z5": oracle_valid[:, :, 0],
        "Z6": oracle_valid[:, :, 1],
        "Z7": oracle_valid[:, :, 2],
    }
    provenance = {
        "seed": int(seed),
        "arm": arm,
        "fold": int(fold),
        "local_feature_sha256": local_sha,
        "local_metadata_sha256": file_sha256(local_metadata_path),
        "spatial_feature_sha256": spatial_sha,
        "spatial_metadata_sha256": file_sha256(spatial_metadata_path),
        "checkpoint_sha256": checkpoint_sha,
        "selection_sha256": selection_sha,
        "patient_order_sha256": local_metadata.get("patient_order_sha256"),
        "selected_epoch": int(local_metadata.get("selected_epoch", -1)),
        "encoder_frozen": bool(spatial_metadata.get("encoder_frozen")),
        "training_performed": bool(spatial_metadata.get("training_performed")),
        "z1_derived_from_z2_without_encoder": True,
        **checkpoint_evidence,
    }
    if provenance["encoder_frozen"] is not True or provenance["training_performed"] is not False:
        raise ValueError("spatial feature metadata does not prove a frozen encoder")
    return FeatureCell(
        seed=int(seed),
        arm=str(arm),
        fold=int(fold),
        patient_ids=targets.patient_ids.copy(),
        splits=expected_splits.copy(),
        representations=representations,
        validity=validity,
        selected_epoch=int(local_metadata["selected_epoch"]),
        provenance=provenance,
    )


@dataclass(frozen=True)
class FittedTargetTransform:
    lower: float
    upper: float
    mean: float
    scale: float
    log1p: bool

    def transform_unscaled(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=np.float64), self.lower, self.upper)
        if self.log1p:
            if np.any(clipped <= -1.0):
                raise ValueError("log1p target contains value <= -1 after train-fitted clipping")
            clipped = np.log1p(clipped)
        return clipped

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (self.transform_unscaled(values) - self.mean) / self.scale

    def inverse_standardized(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = np.asarray(values, dtype=np.float64) * self.scale + self.mean
        natural = np.expm1(transformed) if self.log1p else transformed.copy()
        return natural, transformed


def fit_target_transform(
    values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    log1p: bool,
    quantiles: Sequence[float],
) -> FittedTargetTransform:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(array)
    if int(mask.sum()) < 3:
        raise ValueError("too few outer-train rows for a target transform")
    lower, upper = np.quantile(array[mask], np.asarray(quantiles, dtype=np.float64))
    clipped = np.clip(array[mask], lower, upper)
    if log1p:
        if np.any(clipped <= -1.0):
            raise ValueError("outer-train target is invalid for log1p")
        clipped = np.log1p(clipped)
    mean = float(np.mean(clipped))
    scale = float(np.std(clipped, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("outer-train target is constant or has invalid scale")
    return FittedTargetTransform(
        lower=float(lower),
        upper=float(upper),
        mean=mean,
        scale=scale,
        log1p=bool(log1p),
    )


@dataclass(frozen=True)
class Outcome:
    task_type: str
    target_kind: str
    target: str
    timing: str
    interval: str
    valid: np.ndarray
    probe_y: np.ndarray
    natural_y: np.ndarray
    transformed_y: np.ndarray
    metric_space: str
    raw_transform: FittedTargetTransform | None
    residual_center: float
    residual_scale: float
    residualizer_id: str
    conditional_standardized: np.ndarray | None

    def decode_probe_prediction(
        self,
        prediction: np.ndarray,
        row_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(prediction, dtype=np.float64)
        if self.target_kind == "raw":
            if self.raw_transform is None:
                raise AssertionError("raw outcome is missing its fitted transform")
            return self.raw_transform.inverse_standardized(array)
        residual = array * float(self.residual_scale) + float(self.residual_center)
        if self.raw_transform is None or self.conditional_standardized is None:
            raise AssertionError("residual outcome is missing reconstruction state")
        indices = np.asarray(row_indices, dtype=np.int64)
        conditional = self.conditional_standardized[indices]
        if conditional.shape != residual.shape or not np.isfinite(conditional).all():
            raise ValueError("conditional residual baseline is invalid")
        reconstructed_standardized = conditional + residual
        reconstructed_natural, _ = self.raw_transform.inverse_standardized(
            reconstructed_standardized
        )
        # The second return value remains the residual itself.  Residual rank/R2
        # is evaluated in this Goal-6 transformed-standardized coordinate; only
        # the first return is a conditional reconstruction in natural units.
        return reconstructed_natural, residual


def adjacent_percent_change(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    valid = np.isfinite(start) & np.isfinite(end) & (np.abs(start) > 0.0)
    output = np.full(start.shape, np.nan, dtype=np.float64)
    output[valid] = 100.0 * (end[valid] - start[valid]) / np.abs(start[valid])
    return output, valid


def _transform_audit_row(
    transform: FittedTargetTransform,
    *,
    fold: int,
    task_type: str,
    context: str,
    family: str,
    timing: str,
    interval: str,
    n_train: int,
) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "task_type": task_type,
        "context": context,
        "family": family,
        "timing": timing,
        "interval": interval,
        "fit_scope": "outer_train_only",
        "winsor_lower": transform.lower,
        "winsor_upper": transform.upper,
        "log1p": transform.log1p,
        "train_mean_after_family_transform": transform.mean,
        "train_population_scale": transform.scale,
        "n_train": int(n_train),
    }


def _raw_outcome(
    *,
    values: np.ndarray,
    valid: np.ndarray,
    splits: np.ndarray,
    log1p: bool,
    quantiles: Sequence[float],
    task_type: str,
    target: str,
    timing: str,
    interval: str,
    fold: int,
    transform_rows: list[dict[str, Any]],
) -> Outcome:
    fit_mask = np.asarray(valid, dtype=bool) & (splits == "train")
    transform = fit_target_transform(values, fit_mask, log1p=log1p, quantiles=quantiles)
    probe = np.full(len(values), np.nan, dtype=np.float64)
    transformed = np.full(len(values), np.nan, dtype=np.float64)
    probe[valid] = transform.transform(values[valid])
    transformed[valid] = transform.transform_unscaled(values[valid])
    transform_rows.append(
        _transform_audit_row(
            transform,
            fold=fold,
            task_type=task_type,
            context=f"raw_{target}",
            family=target,
            timing=timing,
            interval=interval,
            n_train=int(fit_mask.sum()),
        )
    )
    return Outcome(
        task_type=task_type,
        target_kind="raw",
        target=target,
        timing=timing,
        interval=interval,
        valid=np.asarray(valid, dtype=bool),
        probe_y=probe,
        natural_y=np.asarray(values, dtype=np.float64),
        transformed_y=transformed,
        metric_space="natural_target",
        raw_transform=transform,
        residual_center=0.0,
        residual_scale=1.0,
        residualizer_id="",
        conditional_standardized=None,
    )


def _residual_outcome(
    *,
    target_values: np.ndarray,
    predictor_values: Sequence[np.ndarray],
    valid: np.ndarray,
    splits: np.ndarray,
    target_log1p: bool,
    predictor_log1p: Sequence[bool],
    predictor_names: Sequence[str],
    quantiles: Sequence[float],
    task_type: str,
    target_kind: str,
    target: str,
    timing: str,
    interval: str,
    fold: int,
    residualizer_alpha: float,
    transform_rows: list[dict[str, Any]],
    residualizer_rows: list[dict[str, Any]],
) -> Outcome:
    valid = np.asarray(valid, dtype=bool)
    fit_mask = valid & (splits == "train")
    target_transform = fit_target_transform(
        target_values, fit_mask, log1p=target_log1p, quantiles=quantiles
    )
    transform_rows.append(
        _transform_audit_row(
            target_transform,
            fold=fold,
            task_type=task_type,
            context=f"{target_kind}_response_{target}",
            family=target,
            timing=timing,
            interval=interval,
            n_train=int(fit_mask.sum()),
        )
    )
    y_standardized = np.full(len(target_values), np.nan, dtype=np.float64)
    y_standardized[valid] = target_transform.transform(target_values[valid])
    predictor_columns: list[np.ndarray] = []
    predictor_transforms: list[FittedTargetTransform] = []
    for values, log1p, name in zip(predictor_values, predictor_log1p, predictor_names, strict=True):
        transform = fit_target_transform(values, fit_mask, log1p=log1p, quantiles=quantiles)
        column = np.full(len(values), np.nan, dtype=np.float64)
        column[valid] = transform.transform(values[valid])
        predictor_columns.append(column)
        predictor_transforms.append(transform)
        transform_rows.append(
            _transform_audit_row(
                transform,
                fold=fold,
                task_type=task_type,
                context=f"{target_kind}_predictor_for_{target}",
                family=name,
                timing=timing,
                interval=interval,
                n_train=int(fit_mask.sum()),
            )
        )
    predictor_matrix = np.column_stack(predictor_columns)
    residualizer = Ridge(alpha=float(residualizer_alpha), fit_intercept=True)
    residualizer.fit(predictor_matrix[fit_mask], y_standardized[fit_mask])
    fitted = np.full(len(target_values), np.nan, dtype=np.float64)
    fitted[valid] = residualizer.predict(predictor_matrix[valid])
    residual = y_standardized - fitted
    residual_center = float(np.mean(residual[fit_mask]))
    residual_scale = float(np.std(residual[fit_mask], ddof=0))
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        residual_scale = 1.0
    probe_y = np.full(len(target_values), np.nan, dtype=np.float64)
    probe_y[valid] = (residual[valid] - residual_center) / residual_scale
    residualizer_id = canonical_sha256(
        {
            "fold": fold,
            "task_type": task_type,
            "target_kind": target_kind,
            "target": target,
            "timing": timing,
            "interval": interval,
            "predictors": list(predictor_names),
            "alpha": float(residualizer_alpha),
        }
    )
    coefficient = np.asarray(residualizer.coef_, dtype=np.float64).reshape(-1)
    residualizer_rows.append(
        {
            "residualizer_id": residualizer_id,
            "fold": int(fold),
            "task_type": task_type,
            "target_kind": target_kind,
            "target": target,
            "timing": timing,
            "interval": interval,
            "predictors": "+".join(predictor_names),
            "model": "Ridge",
            "alpha": float(residualizer_alpha),
            "fit_scope": "outer_train_only",
            "target_space": "Goal6_train_winsorized_family_transformed_standardized",
            "n_train": int(fit_mask.sum()),
            "intercept": float(np.asarray(residualizer.intercept_).reshape(-1)[0]),
            "coefficient_json": json.dumps(coefficient.tolist(), separators=(",", ":")),
            "residual_train_mean": residual_center,
            "residual_train_population_scale": residual_scale,
        }
    )
    return Outcome(
        task_type=task_type,
        target_kind=target_kind,
        target=target,
        timing=timing,
        interval=interval,
        valid=valid,
        probe_y=probe_y,
        natural_y=np.asarray(target_values, dtype=np.float64).copy(),
        transformed_y=residual.copy(),
        metric_space="goal6_transformed_standardized_residual",
        raw_transform=target_transform,
        residual_center=residual_center,
        residual_scale=residual_scale,
        residualizer_id=residualizer_id,
        conditional_standardized=fitted.copy(),
    )


def build_outcomes(
    config: Mapping[str, Any],
    targets: TargetDataset,
    fold_splits: Mapping[int, np.ndarray],
) -> tuple[
    Mapping[tuple[int, str], tuple[Outcome, ...]],
    Mapping[tuple[int, str], tuple[Outcome, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    quantiles = tuple(float(value) for value in config["target_transforms"]["winsor_quantiles"])
    log_families = frozenset(config["target_transforms"]["log1p_families"])
    residualizer_alpha = float(config["residualization"]["alpha"])
    static: dict[tuple[int, str], tuple[Outcome, ...]] = {}
    dynamic: dict[tuple[int, str], tuple[Outcome, ...]] = {}
    transform_rows: list[dict[str, Any]] = []
    residualizer_rows: list[dict[str, Any]] = []
    all_valid = np.ones(len(targets.patient_ids), dtype=bool)

    for fold, splits in fold_splits.items():
        for visit_index, timing in enumerate(VISITS):
            outcomes: list[Outcome] = []
            for family in FAMILIES:
                outcomes.append(
                    _raw_outcome(
                        values=targets.values[family][:, visit_index],
                        valid=all_valid,
                        splits=splits,
                        log1p=family in log_families,
                        quantiles=quantiles,
                        task_type="static",
                        target=family,
                        timing=timing,
                        interval="",
                        fold=fold,
                        transform_rows=transform_rows,
                    )
                )
            ftv_values = targets.values["FTV"][:, visit_index]
            ld_values = targets.values["LD"][:, visit_index]
            for family in PRIMARY_RESIDUAL_TARGETS:
                outcomes.append(
                    _residual_outcome(
                        target_values=targets.values[family][:, visit_index],
                        predictor_values=(ftv_values,),
                        valid=all_valid,
                        splits=splits,
                        target_log1p=family in log_families,
                        predictor_log1p=(True,),
                        predictor_names=("FTV",),
                        quantiles=quantiles,
                        task_type="static",
                        target_kind="residual_ftv",
                        target=family,
                        timing=timing,
                        interval="",
                        fold=fold,
                        residualizer_alpha=residualizer_alpha,
                        transform_rows=transform_rows,
                        residualizer_rows=residualizer_rows,
                    )
                )
            for family in SECONDARY_RESIDUAL_TARGETS:
                outcomes.append(
                    _residual_outcome(
                        target_values=targets.values[family][:, visit_index],
                        predictor_values=(ftv_values, ld_values),
                        valid=all_valid,
                        splits=splits,
                        target_log1p=family in log_families,
                        predictor_log1p=(True, True),
                        predictor_names=("FTV", "LD"),
                        quantiles=quantiles,
                        task_type="static",
                        target_kind="residual_ftv_ld",
                        target=family,
                        timing=timing,
                        interval="",
                        fold=fold,
                        residualizer_alpha=residualizer_alpha,
                        transform_rows=transform_rows,
                        residualizer_rows=residualizer_rows,
                    )
                )
            static[(fold, timing)] = tuple(outcomes)

        for interval_index, interval in enumerate(INTERVALS):
            changes: dict[str, np.ndarray] = {}
            validities: dict[str, np.ndarray] = {}
            for family in FAMILIES:
                change, valid = adjacent_percent_change(
                    targets.values[family][:, interval_index],
                    targets.values[family][:, interval_index + 1],
                )
                changes[family] = change
                validities[family] = valid
            outcomes = []
            for family in FAMILIES:
                outcomes.append(
                    _raw_outcome(
                        values=changes[family],
                        valid=validities[family],
                        splits=splits,
                        log1p=False,
                        quantiles=quantiles,
                        task_type="dynamic",
                        target=family,
                        timing="",
                        interval=interval,
                        fold=fold,
                        transform_rows=transform_rows,
                    )
                )
            for family in PRIMARY_RESIDUAL_TARGETS:
                valid = validities[family] & validities["FTV"]
                outcomes.append(
                    _residual_outcome(
                        target_values=changes[family],
                        predictor_values=(changes["FTV"],),
                        valid=valid,
                        splits=splits,
                        target_log1p=False,
                        predictor_log1p=(False,),
                        predictor_names=("delta_FTV",),
                        quantiles=quantiles,
                        task_type="dynamic",
                        target_kind="residual_ftv",
                        target=family,
                        timing="",
                        interval=interval,
                        fold=fold,
                        residualizer_alpha=residualizer_alpha,
                        transform_rows=transform_rows,
                        residualizer_rows=residualizer_rows,
                    )
                )
            for family in SECONDARY_RESIDUAL_TARGETS:
                valid = validities[family] & validities["FTV"] & validities["LD"]
                outcomes.append(
                    _residual_outcome(
                        target_values=changes[family],
                        predictor_values=(changes["FTV"], changes["LD"]),
                        valid=valid,
                        splits=splits,
                        target_log1p=False,
                        predictor_log1p=(False, False),
                        predictor_names=("delta_FTV", "delta_LD"),
                        quantiles=quantiles,
                        task_type="dynamic",
                        target_kind="residual_ftv_ld",
                        target=family,
                        timing="",
                        interval=interval,
                        fold=fold,
                        residualizer_alpha=residualizer_alpha,
                        transform_rows=transform_rows,
                        residualizer_rows=residualizer_rows,
                    )
                )
            dynamic[(fold, interval)] = tuple(outcomes)
    return static, dynamic, transform_rows, residualizer_rows


def regression_metrics(
    y_natural: np.ndarray,
    prediction_natural: np.ndarray,
    y_transformed: np.ndarray,
    prediction_transformed: np.ndarray,
) -> dict[str, float]:
    natural = np.asarray(y_natural, dtype=np.float64)
    predicted = np.asarray(prediction_natural, dtype=np.float64)
    transformed = np.asarray(y_transformed, dtype=np.float64)
    predicted_transformed = np.asarray(prediction_transformed, dtype=np.float64)
    if not (
        natural.ndim == predicted.ndim == transformed.ndim == predicted_transformed.ndim == 1
        and len(natural) == len(predicted) == len(transformed) == len(predicted_transformed)
        and len(natural) >= 2
    ):
        raise ValueError("metric arrays must be aligned nontrivial vectors")
    if not all(np.isfinite(array).all() for array in (natural, predicted, transformed, predicted_transformed)):
        raise FloatingPointError("metric arrays contain NaN or Inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        natural_spearman = float(spearmanr(natural, predicted).statistic)
        natural_pearson = float(pearsonr(natural, predicted).statistic)
        transformed_spearman = float(
            spearmanr(transformed, predicted_transformed).statistic
        )
        transformed_pearson = float(
            pearsonr(transformed, predicted_transformed).statistic
        )
    natural_sst = float(np.sum((natural - natural.mean()) ** 2))
    transformed_sst = float(np.sum((transformed - transformed.mean()) ** 2))
    natural_r2 = float("nan") if natural_sst <= 0 else 1.0 - float(np.sum((natural - predicted) ** 2)) / natural_sst
    transformed_r2 = (
        float("nan")
        if transformed_sst <= 0
        else 1.0 - float(np.sum((transformed - predicted_transformed) ** 2)) / transformed_sst
    )
    target_variance = float(np.var(natural, ddof=0))
    prediction_variance = float(np.var(predicted, ddof=0))
    transformed_target_variance = float(np.var(transformed, ddof=0))
    transformed_prediction_variance = float(np.var(predicted_transformed, ddof=0))
    variance_ratio = (
        float("nan") if target_variance <= 0 else prediction_variance / target_variance
    )
    transformed_variance_ratio = (
        float("nan")
        if transformed_target_variance <= 0
        else transformed_prediction_variance / transformed_target_variance
    )
    # Conventional calibration slope regresses observed target on prediction:
    # Cov(target,prediction) / Var(prediction), with ideal value one.
    if prediction_variance <= 0:
        calibration_slope = float("nan")
        calibration_intercept = float("nan")
    else:
        calibration_slope = float(
            np.mean((predicted - predicted.mean()) * (natural - natural.mean()))
            / prediction_variance
        )
        calibration_intercept = float(
            natural.mean() - calibration_slope * predicted.mean()
        )
    if transformed_prediction_variance <= 0:
        transformed_calibration_slope = float("nan")
        transformed_calibration_intercept = float("nan")
    else:
        transformed_calibration_slope = float(
            np.mean(
                (predicted_transformed - predicted_transformed.mean())
                * (transformed - transformed.mean())
            )
            / transformed_prediction_variance
        )
        transformed_calibration_intercept = float(
            transformed.mean()
            - transformed_calibration_slope * predicted_transformed.mean()
        )
    return {
        "natural_spearman": natural_spearman,
        "natural_pearson": natural_pearson,
        "transformed_spearman": transformed_spearman,
        "transformed_pearson": transformed_pearson,
        "natural_r2": natural_r2,
        "transformed_r2": transformed_r2,
        "natural_rmse": float(np.sqrt(np.mean((natural - predicted) ** 2))),
        "natural_mae": float(np.mean(np.abs(natural - predicted))),
        "transformed_rmse": float(
            np.sqrt(np.mean((transformed - predicted_transformed) ** 2))
        ),
        "transformed_mae": float(
            np.mean(np.abs(transformed - predicted_transformed))
        ),
        "natural_prediction_target_variance_ratio": variance_ratio,
        "transformed_prediction_target_variance_ratio": transformed_variance_ratio,
        "natural_calibration_slope": calibration_slope,
        "natural_calibration_intercept": calibration_intercept,
        "transformed_calibration_slope": transformed_calibration_slope,
        "transformed_calibration_intercept": transformed_calibration_intercept,
    }


def present_metrics(
    metrics: Mapping[str, float],
    *,
    residual: bool,
) -> dict[str, float | str]:
    """Expose unambiguous raw and residual metric names.

    For residual outcomes, rank and transformed R2 refer to epsilon_y.  Natural
    quantities refer only to the conditional reconstruction
    ridge_train(FTV[*]) + predicted epsilon_y; they are never described as a
    natural-unit residual metric.
    """

    nan = float("nan")
    return {
        "spearman": metrics[
            "transformed_spearman" if residual else "natural_spearman"
        ],
        "pearson": metrics[
            "transformed_pearson" if residual else "natural_pearson"
        ],
        "natural_r2": nan if residual else metrics["natural_r2"],
        "transformed_r2": metrics["transformed_r2"],
        "rmse": nan if residual else metrics["natural_rmse"],
        "mae": nan if residual else metrics["natural_mae"],
        "prediction_target_variance_ratio": metrics[
            "transformed_prediction_target_variance_ratio"
            if residual
            else "natural_prediction_target_variance_ratio"
        ],
        "calibration_slope": metrics[
            "transformed_calibration_slope"
            if residual
            else "natural_calibration_slope"
        ],
        "calibration_intercept": metrics[
            "transformed_calibration_intercept"
            if residual
            else "natural_calibration_intercept"
        ],
        "residual_spearman": metrics["transformed_spearman"] if residual else nan,
        "residual_pearson": metrics["transformed_pearson"] if residual else nan,
        "residual_transformed_r2": metrics["transformed_r2"] if residual else nan,
        "residual_rmse": metrics["transformed_rmse"] if residual else nan,
        "residual_mae": metrics["transformed_mae"] if residual else nan,
        "reconstructed_natural_r2": metrics["natural_r2"] if residual else nan,
        "reconstructed_natural_rmse": metrics["natural_rmse"] if residual else nan,
        "reconstructed_natural_mae": metrics["natural_mae"] if residual else nan,
        "reconstructed_prediction_target_variance_ratio": (
            metrics["natural_prediction_target_variance_ratio"] if residual else nan
        ),
        "reconstructed_calibration_slope": (
            metrics["natural_calibration_slope"] if residual else nan
        ),
        "reconstructed_calibration_intercept": (
            metrics["natural_calibration_intercept"] if residual else nan
        ),
        "natural_spearman": metrics["natural_spearman"],
        "natural_pearson": metrics["natural_pearson"],
        "transformed_spearman": metrics["transformed_spearman"],
        "transformed_pearson": metrics["transformed_pearson"],
        "natural_metric_interpretation": (
            "conditional_target_reconstruction" if residual else "raw_target"
        ),
    }


PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "seed",
    "arm",
    "representation",
    "matched_reference_for",
    "task_type",
    "target_definition",
    "target_kind",
    "target",
    "timing",
    "interval",
    "input_variant",
    "metric_space",
    "feature_dim",
    "selected_alpha",
    "y_true_natural",
    "y_pred_natural",
    "y_true_transformed",
    "y_pred_transformed",
)


class PrivatePredictionWriter:
    def __init__(self, path: str | Path) -> None:
        self.output = Path(path)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.output.parent, 0o700)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent
        )
        os.close(descriptor)
        self.temporary = Path(temporary)
        os.chmod(self.temporary, 0o600)
        self.stream = gzip.open(self.temporary, "wt", encoding="utf-8", newline="", compresslevel=6)
        self.writer = csv.DictWriter(self.stream, fieldnames=PREDICTION_COLUMNS)
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow({column: row[column] for column in PREDICTION_COLUMNS})
        self.rows += 1

    def close(self) -> None:
        self.stream.close()
        self.temporary.replace(self.output)
        self.output.chmod(0o600)

    def abort(self) -> None:
        try:
            self.stream.close()
        finally:
            self.temporary.unlink(missing_ok=True)


class OOFAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def add(
        self,
        metadata: Mapping[str, Any],
        *,
        patient_ids: np.ndarray,
        natural: np.ndarray,
        predicted_natural: np.ndarray,
        transformed: np.ndarray,
        predicted_transformed: np.ndarray,
    ) -> None:
        identity = dict(metadata)
        fold = int(identity.pop("fold_for_accumulator"))
        identifiers = np.asarray(patient_ids, dtype=str)
        if identifiers.ndim != 1 or len(np.unique(identifiers)) != len(identifiers):
            raise ValueError("an OOF fold chunk contains duplicate patient identifiers")
        key = canonical_sha256(identity)
        if key not in self.values:
            self.values[key] = {
                "metadata": identity,
                "natural": [],
                "predicted_natural": [],
                "transformed": [],
                "predicted_transformed": [],
                "folds": set(),
                "fold_chunks": [],
                "patient_ids": set(),
            }
        record = self.values[key]
        if record["metadata"] != identity:
            raise AssertionError("OOF metadata hash collision")
        if fold in record["folds"]:
            raise ValueError("an OOF endpoint contains a fold more than once")
        if record["patient_ids"].intersection(identifiers.tolist()):
            raise ValueError("an OOF endpoint contains a patient more than once")
        record["patient_ids"].update(identifiers.tolist())
        record["natural"].append(np.asarray(natural, dtype=np.float64))
        record["predicted_natural"].append(np.asarray(predicted_natural, dtype=np.float64))
        record["transformed"].append(np.asarray(transformed, dtype=np.float64))
        record["predicted_transformed"].append(np.asarray(predicted_transformed, dtype=np.float64))
        record["folds"].add(fold)
        record["fold_chunks"].append(
            (
                fold,
                np.asarray(natural, dtype=np.float64),
                np.asarray(predicted_natural, dtype=np.float64),
                np.asarray(transformed, dtype=np.float64),
                np.asarray(predicted_transformed, dtype=np.float64),
                identifiers.copy(),
            )
        )

    def metrics_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in self.values.values():
            metadata = dict(record["metadata"])
            natural = np.concatenate(record["natural"])
            predicted_natural = np.concatenate(record["predicted_natural"])
            transformed = np.concatenate(record["transformed"])
            predicted_transformed = np.concatenate(record["predicted_transformed"])
            base_metrics = regression_metrics(
                natural,
                predicted_natural,
                transformed,
                predicted_transformed,
            )
            fold_transformed_r2: list[float] = []
            fold_weights: list[int] = []
            fold_transformed_metrics: list[dict[str, float]] = []
            observed_folds: set[int] = set()
            for fold, fold_natural, fold_predicted_natural, fold_transformed, fold_predicted_transformed, _fold_ids in record["fold_chunks"]:
                if fold in observed_folds:
                    raise ValueError("an OOF endpoint contains duplicate fold chunks")
                observed_folds.add(int(fold))
                fold_base = regression_metrics(
                    fold_natural,
                    fold_predicted_natural,
                    fold_transformed,
                    fold_predicted_transformed,
                )
                fold_transformed_r2.append(float(fold_base["transformed_r2"]))
                fold_weights.append(len(fold_transformed))
                fold_transformed_metrics.append(fold_base)
            if any(not np.isfinite(value) for value in fold_transformed_r2):
                weighted_transformed_r2 = float("nan")
            else:
                weighted_transformed_r2 = float(
                    np.average(fold_transformed_r2, weights=fold_weights)
                )
            base_metrics["transformed_r2"] = weighted_transformed_r2
            if metadata["target_kind"] != "raw":
                for metric_name in (
                    "transformed_spearman",
                    "transformed_pearson",
                    "transformed_mae",
                    "transformed_prediction_target_variance_ratio",
                    "transformed_calibration_slope",
                    "transformed_calibration_intercept",
                ):
                    values = [
                        float(metrics[metric_name])
                        for metrics in fold_transformed_metrics
                    ]
                    base_metrics[metric_name] = (
                        float(np.average(values, weights=fold_weights))
                        if all(np.isfinite(value) for value in values)
                        else float("nan")
                    )
                fold_mse = [
                    float(metrics["transformed_rmse"]) ** 2
                    for metrics in fold_transformed_metrics
                ]
                base_metrics["transformed_rmse"] = (
                    float(np.sqrt(np.average(fold_mse, weights=fold_weights)))
                    if all(np.isfinite(value) for value in fold_mse)
                    else float("nan")
                )
            row = {
                **metadata,
                "n": len(natural),
                "n_folds": len(record["folds"]),
                "eligible_patient_set_sha256": ordered_sha256(
                    sorted(record["patient_ids"])
                ),
                "oof_identity_sha256": ordered_sha256(
                    sorted(
                        f"{patient_id}|{fold}"
                        for fold, _, _, _, _, fold_ids in record["fold_chunks"]
                        for patient_id in fold_ids
                    )
                ),
                "transformed_r2_aggregation": "outer_test_n_weighted_fold_r2",
                "rank_aggregation": (
                    "outer_test_n_weighted_fold_residual_metric"
                    if metadata["target_kind"] != "raw"
                    else "pooled_oof_natural_target"
                ),
                **present_metrics(
                    base_metrics,
                    residual=metadata["target_kind"] != "raw",
                ),
            }
            rows.append(row)
        return pd.DataFrame(rows)


def _ridge_path_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    alphas: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Exact centered dual ridge path with one eigendecomposition per mask.

    Returns validation predictions for every alpha, selected-test helper duals,
    and the outer-train scaler parameters.  No test target enters this function.
    """

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0, ddof=0)
    constant_columns = int(np.sum(~np.isfinite(feature_scale) | (feature_scale <= 0.0)))
    feature_scale[~np.isfinite(feature_scale) | (feature_scale <= 0.0)] = 1.0
    train_scaled = (x_train - feature_mean) / feature_scale
    validation_scaled = (x_validation - feature_mean) / feature_scale
    test_scaled = (x_test - feature_mean) / feature_scale
    x_center = np.mean(train_scaled, axis=0)
    y_center = np.mean(y_train, axis=0)
    centered_train = train_scaled - x_center
    centered_y = y_train - y_center
    kernel = centered_train @ centered_train.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ centered_y
    validation_cross = (validation_scaled - x_center) @ centered_train.T
    test_cross = (test_scaled - x_center) @ centered_train.T
    validation_predictions: list[np.ndarray] = []
    duals: list[np.ndarray] = []
    for alpha in alphas:
        dual = eigenvectors @ (projected_y / (eigenvalues[:, None] + float(alpha)))
        duals.append(dual)
        validation_predictions.append(y_center + validation_cross @ dual)
    return (
        np.stack(validation_predictions, axis=0),
        np.stack(duals, axis=0),
        test_cross,
        y_center,
        constant_columns,
    )


def probe_outcome_batch(
    *,
    x: np.ndarray,
    feature_valid: np.ndarray,
    outcomes: Sequence[Outcome],
    cell: FeatureCell,
    representation: str,
    matched_reference_for: str,
    input_variant: str,
    config: Mapping[str, Any],
    writer: PrivatePredictionWriter,
    accumulator: OOFAccumulator,
    selection_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(cell.patient_ids):
        raise ValueError("probe feature matrix shape is invalid")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("probe feature matrix contains NaN or Inf")
    feature_valid = np.asarray(feature_valid, dtype=bool)
    if feature_valid.shape != (len(matrix),):
        raise ValueError("feature validity shape is invalid")
    groups: dict[bytes, list[Outcome]] = {}
    masks: dict[bytes, np.ndarray] = {}
    for outcome in outcomes:
        mask = feature_valid & outcome.valid
        key = mask.tobytes()
        groups.setdefault(key, []).append(outcome)
        masks[key] = mask

    alphas = tuple(float(alpha) for alpha in config["probe"]["alphas"])
    for key, grouped_outcomes in groups.items():
        eligible = masks[key]
        train_mask = eligible & (cell.splits == "train")
        validation_mask = eligible & (cell.splits == "val")
        test_mask = eligible & (cell.splits == "test")
        n_train = int(train_mask.sum())
        n_validation = int(validation_mask.sum())
        n_test = int(test_mask.sum())
        if min(n_train, n_validation, n_test) < 3:
            raise ValueError("too few eligible rows in a probe split")
        y_train = np.column_stack([outcome.probe_y[train_mask] for outcome in grouped_outcomes])
        if not np.isfinite(y_train).all():
            raise FloatingPointError("probe target contains NaN or Inf on outer train")
        validation_predictions, duals, test_cross, y_center, constant_columns = _ridge_path_predictions(
            matrix[train_mask],
            y_train,
            matrix[validation_mask],
            matrix[test_mask],
            alphas,
        )
        validation_true = np.column_stack(
            [outcome.probe_y[validation_mask] for outcome in grouped_outcomes]
        )
        validation_mse = np.mean(
            (validation_predictions - validation_true[None, :, :]) ** 2,
            axis=1,
        )
        selected_indices = np.argmin(validation_mse, axis=0)
        for column, outcome in enumerate(grouped_outcomes):
            selected_index = int(selected_indices[column])
            selected_alpha = alphas[selected_index]
            predicted_probe = y_center[column] + test_cross @ duals[selected_index, :, column]
            if not np.isfinite(predicted_probe).all():
                raise FloatingPointError("selected ridge emitted a non-finite test prediction")
            test_indices = np.flatnonzero(test_mask)
            predicted_natural, predicted_transformed = outcome.decode_probe_prediction(
                predicted_probe,
                test_indices,
            )
            true_natural = outcome.natural_y[test_mask]
            true_transformed = outcome.transformed_y[test_mask]
            metadata = {
                "seed": cell.seed,
                "arm": cell.arm,
                "representation": representation,
                "matched_reference_for": matched_reference_for,
                "task_type": outcome.task_type,
                "target_definition": (
                    "adjacent_percent_change_new_extension"
                    if outcome.task_type == "dynamic"
                    else "goal6_workbook_endpoint"
                ),
                "target_kind": outcome.target_kind,
                "target": outcome.target,
                "timing": outcome.timing,
                "interval": outcome.interval,
                "input_variant": input_variant,
                "metric_space": outcome.metric_space,
                "feature_dim": int(matrix.shape[1]),
            }
            fold_metrics = present_metrics(
                regression_metrics(
                    true_natural,
                    predicted_natural,
                    true_transformed,
                    predicted_transformed,
                ),
                residual=outcome.target_kind != "raw",
            )
            fold_metric_rows.append(
                {
                    **metadata,
                    "fold": cell.fold,
                    "n": n_test,
                    **fold_metrics,
                }
            )
            validation_scores = {
                format(alpha, ".10g"): float(validation_mse[index, column])
                for index, alpha in enumerate(alphas)
            }
            selection_rows.append(
                {
                    **metadata,
                    "fold": cell.fold,
                    "n_train": n_train,
                    "n_validation": n_validation,
                    "n_test": n_test,
                    "selected_alpha": selected_alpha,
                    "selected_validation_mse_standardized": float(validation_mse[selected_index, column]),
                    "alpha_validation_mse_json": json.dumps(validation_scores, sort_keys=True, separators=(",", ":")),
                    "feature_scaler": "outer_train_StandardScaler_population_variance",
                    "feature_constant_columns": constant_columns,
                    "ridge_solver": "exact_centered_dual_eigendecomposition",
                    "ridge_fit_intercept": True,
                    "alpha_tie_break": "smaller_alpha",
                    "test_used_for_scaler": False,
                    "test_used_for_alpha_selection": False,
                    "test_predict_call_count": 1,
                    "residualizer_id": outcome.residualizer_id,
                }
            )
            coverage_rows.append(
                {
                    **metadata,
                    "fold": cell.fold,
                    "n_train": n_train,
                    "n_validation": n_validation,
                    "n_test": n_test,
                    "feature_valid_total": int(feature_valid.sum()),
                    "target_valid_total": int(outcome.valid.sum()),
                    "joint_valid_total": int(eligible.sum()),
                }
            )
            for local_index, patient_index in enumerate(test_indices):
                writer.write(
                    {
                        "patient_id": str(cell.patient_ids[patient_index]),
                        "fold": cell.fold,
                        **metadata,
                        "selected_alpha": selected_alpha,
                        "y_true_natural": float(true_natural[local_index]),
                        "y_pred_natural": float(predicted_natural[local_index]),
                        "y_true_transformed": float(true_transformed[local_index]),
                        "y_pred_transformed": float(predicted_transformed[local_index]),
                    }
                )
            accumulator.add(
                {**metadata, "fold_for_accumulator": cell.fold},
                patient_ids=cell.patient_ids[test_indices],
                natural=true_natural,
                predicted_natural=predicted_natural,
                transformed=true_transformed,
                predicted_transformed=predicted_transformed,
            )


def static_views(cell: FeatureCell, visit_index: int) -> Mapping[str, tuple[np.ndarray, np.ndarray, str]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for representation in MAIN_REPRESENTATIONS:
        output[representation] = (
            cell.representations[representation][:, visit_index],
            cell.validity[representation][:, visit_index],
            "",
        )
    for oracle, reference in ORACLE_TO_MATCHED.items():
        output[reference] = (
            cell.representations["Z4"][:, visit_index],
            cell.validity[oracle][:, visit_index],
            oracle,
        )
    return output


def dynamic_views(
    cell: FeatureCell,
    interval_index: int,
    input_variant: str,
    *,
    include_matched_references: bool,
) -> Mapping[str, tuple[np.ndarray, np.ndarray, str]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for representation in MAIN_REPRESENTATIONS:
        features = cell.representations[representation]
        start = features[:, interval_index]
        end = features[:, interval_index + 1]
        difference = end - start
        matrix = difference if input_variant == "difference" else np.concatenate((start, end, difference), axis=1)
        valid = cell.validity[representation][:, interval_index] & cell.validity[representation][:, interval_index + 1]
        output[representation] = (matrix, valid, "")
    if include_matched_references:
        features = cell.representations["Z4"]
        start = features[:, interval_index]
        end = features[:, interval_index + 1]
        difference = end - start
        matrix = difference if input_variant == "difference" else np.concatenate((start, end, difference), axis=1)
        for oracle, reference in ORACLE_TO_MATCHED.items():
            valid = cell.validity[oracle][:, interval_index] & cell.validity[oracle][:, interval_index + 1]
            output[reference] = (matrix, valid, oracle)
    return output


def representation_pair_comparisons(oof: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("Z2", "Z1", "PREPROJECTOR_MINUS_PROJECTED"),
        ("Z3", "Z2", "SPATIAL_MEAN_MINUS_PREPROJECTOR"),
        ("Z4", "Z3", "MEAN_STD_MINUS_MEAN"),
    )
    identity = [
        "seed",
        "arm",
        "task_type",
        "target_definition",
        "target_kind",
        "target",
        "timing",
        "interval",
        "input_variant",
        "metric_space",
    ]
    metrics = [
        "spearman",
        "pearson",
        "natural_r2",
        "transformed_r2",
        "residual_spearman",
        "residual_transformed_r2",
        "reconstructed_natural_r2",
        "rmse",
        "mae",
        "reconstructed_natural_rmse",
        "reconstructed_natural_mae",
        "prediction_target_variance_ratio",
        "calibration_slope",
    ]
    rows: list[pd.DataFrame] = []
    main = oof.loc[oof["representation"].isin(MAIN_REPRESENTATIONS)].copy()
    for candidate, reference, comparison in pairs:
        left = main.loc[main["representation"] == candidate, identity + ["n", *metrics]].copy()
        right = main.loc[main["representation"] == reference, identity + ["n", *metrics]].copy()
        merged = left.merge(right, on=identity, suffixes=("_candidate", "_reference"), validate="one_to_one")
        merged.insert(0, "comparison", comparison)
        merged.insert(1, "candidate", candidate)
        merged.insert(2, "reference", reference)
        for metric in metrics:
            merged[f"delta_{metric}"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def localization_comparisons(oof: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "seed",
        "arm",
        "task_type",
        "target_definition",
        "target_kind",
        "target",
        "timing",
        "interval",
        "input_variant",
        "metric_space",
    ]
    metrics = [
        "spearman",
        "pearson",
        "natural_r2",
        "transformed_r2",
        "residual_spearman",
        "residual_transformed_r2",
        "reconstructed_natural_r2",
        "rmse",
        "mae",
        "reconstructed_natural_rmse",
        "reconstructed_natural_mae",
        "prediction_target_variance_ratio",
        "calibration_slope",
    ]
    rows: list[pd.DataFrame] = []
    for oracle, matched in ORACLE_TO_MATCHED.items():
        candidate = oof.loc[oof["representation"] == oracle, identity + ["n", *metrics]].copy()
        reference = oof.loc[oof["representation"] == matched, identity + ["n", *metrics]].copy()
        merged = candidate.merge(reference, on=identity, suffixes=("_oracle", "_full_local_matched"), validate="one_to_one")
        if not np.array_equal(merged["n_oracle"].to_numpy(), merged["n_full_local_matched"].to_numpy()):
            raise ValueError("oracle and matched Z4 comparison populations differ")
        merged.insert(0, "oracle_representation", oracle)
        merged.insert(1, "matched_reference", matched)
        for metric in metrics:
            merged[f"delta_{metric}"] = merged[f"{metric}_oracle"] - merged[f"{metric}_full_local_matched"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def public_artifact_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.relative_to(AUDIT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "contains_patient_identifiers": False,
            }
        )
    return rows


__all__ = [
    "AUDIT_ROOT",
    "FAMILIES",
    "INTERVALS",
    "MAIN_REPRESENTATIONS",
    "OOFAccumulator",
    "ORACLE_TO_MATCHED",
    "PrivatePredictionWriter",
    "REPO_ROOT",
    "VISITS",
    "atomic_csv",
    "atomic_json",
    "authenticate",
    "build_outcomes",
    "canonical_sha256",
    "dynamic_views",
    "file_sha256",
    "load_config",
    "load_feature_cell",
    "load_fold_splits",
    "load_targets",
    "localization_comparisons",
    "ordered_sha256",
    "probe_outcome_batch",
    "public_artifact_manifest",
    "regression_metrics",
    "representation_pair_comparisons",
    "resolve_path",
    "static_views",
]
