"""Frozen data and experiment contracts for the conditional pCR ceiling.

This module is intentionally fail closed.  It verifies the public experiment
configuration, resolves private inputs from one explicit private repository
root, verifies every manifest hash before parsing it, and then proves that the
clinical, fold, and C1B cache populations align on the same 808 patients.
Patient identifiers remain in private in-memory objects and are never included
in audit summaries or exception messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "experiment.json"

ARMS = ("B0", "B1", "B2", "B3")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2026, 3026)
OPTIONAL_SEED = 4026
TIMINGS = ("T0", "T0_T1", "T0_T2", "T0_T3")
PRIMARY_TIMINGS = ("T0", "T0_T1", "T0_T2")
SUPPLEMENTARY_TIMING = "T0_T3"
MATCHING_FIELDS = ("label_hr", "label_her2", "arm")
FULL_COHORT_NAME = "full_808"
FULL_COHORT_SIZE = 808
FTV_COHORT_NAME = "ftv_complete_375"
FTV_COHORT_SIZE = 375

# This is the immutable config supplied with the experiment scaffold.  Copies
# may be validated semantically by passing ``expected_sha256=None`` explicitly;
# the canonical path is always checked against this digest.
LOCKED_CONFIG_SHA256 = (
    "21371b958a31a7f84813ad8c08664126c1664a5f8bf7200a9bcbcefdbcbfc554"
)
LOCKED_CONFIG_CANONICAL_SHA256 = (
    "f34140aa2b6618c0934bc6cf5543835bda6222b0c40f5731e0fae6ddb827b407"
)
LOCKED_STAGE_B_DATA_CONTRACT_SHA256 = (
    "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
)
LOCKED_CLINICAL_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
)
LOCKED_FOLD_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
LOCKED_FTV_SHA256 = (
    "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
)
CONFIRMATION_DIRECTORY = "local_response_state_multiseed_confirmation"

CLINICAL_COLUMNS = (
    "patient_id",
    "label_pcr",
    "label_hr",
    "label_her2",
    "arm",
)
FOLD_COLUMNS = ("patient_id", "fold", "split", "label_pcr")
CACHE_COLUMNS = (
    "patient_id",
    "cache_path",
    "cache_sha256",
    "cache_size_bytes",
    "cache_mtime_ns",
    "input_kind",
)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 without materializing a private artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip()
    if len(digest) != 64 or digest.lower() != digest or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _verified_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    if not source.is_file():
        raise ValueError(f"{label} must resolve to a regular file")
    observed = file_sha256(source)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return source


def _exact_keys(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if set(value) != set(expected):
        raise ValueError(
            f"{label} fields drifted: expected {sorted(expected)}, got {sorted(value)}"
        )
    return value


def _at(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"configuration lacks required field {dotted}")
        value = value[component]
    return value


def _expect(payload: Mapping[str, Any], dotted: str, expected: Any) -> None:
    observed = _at(payload, dotted)
    if observed != expected:
        raise ValueError(
            f"configuration field {dotted} is frozen to {expected!r}; got {observed!r}"
        )


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every scientific factor consumed by the core package."""

    top = (
        "schema_version",
        "experiment",
        "branch",
        "parent_commit",
        "scientific_status",
        "input",
        "paths",
        "populations",
        "folds",
        "seeds",
        "optional_seed",
        "timings",
        "primary_timings",
        "supplementary_timing",
        "matching",
        "arms",
        "projection_head",
        "training",
        "downstream",
        "metrics",
        "calibration",
        "bootstrap",
        "success_gates",
        "reporting_boundary",
    )
    _exact_keys(payload, top, "experiment configuration")
    frozen: Mapping[str, Any] = {
        "schema_version": 1,
        "experiment": "conditional_pcr_contrastive_ceiling",
        "scientific_status": "oracle_supervised_representation_ceiling_not_world_model",
        "input.modality": "C1B-H DCE7",
        "input.shape_bvcdhw": [4, 7, 112, 176, 160],
        "input.base_architecture": "confirmed LOCAL",
        "input.base_arm": "LOCAL3",
        "input.base_state_dim": 192,
        "input.projection_dim": 64,
        "populations.full_808": FULL_COHORT_SIZE,
        "populations.ftv_complete_375": FTV_COHORT_SIZE,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "optional_seed": OPTIONAL_SEED,
        "timings": list(TIMINGS),
        "primary_timings": list(PRIMARY_TIMINGS),
        "supplementary_timing": SUPPLEMENTARY_TIMING,
        "matching.fields": list(MATCHING_FIELDS),
        "matching.scope": "outer_train_only",
        "matching.positive": "same_pcr_within_exact_stratum",
        "matching.negative": "opposite_pcr_within_exact_stratum",
        "matching.unmatched_fallback": False,
        "matching.test_patients_allowed": False,
        "projection_head.architecture": "Linear(192,128)-GELU-LayerNorm(128)-Linear(128,64)",
        "projection_head.temperature": 0.1,
        "projection_head.longitudinal_training": "concatenate projected observed visits within each prefix",
        "training.B1.full_train_contrastive_batch": True,
        "training.B2.encoder_learning_rate": 0.00005,
        "training.B3.encoder_learning_rate": 0.00001,
        "training.B2.head_learning_rate": 0.0005,
        "training.B3.head_learning_rate": 0.0005,
        "training.physical_patient_batch_max": 4,
        "training.anchors_per_epoch_strategy": "all_eligible_anchors_exactly_once_per_epoch",
        "training.test_labels_forbidden": True,
        "downstream.mri_prefix": "literal observed-state concatenation",
        "downstream.pca_fit_scope": "outer_train_only",
        "downstream.classifier": "fold_isolated_L2_logistic_regression",
        "downstream.clinical_matching_fields_are_not_mri_inputs": True,
        "bootstrap.draws": 5000,
        "bootstrap.unit": "patient",
        "bootstrap.stratify_by": "outer_fold",
    }
    for dotted, expected in frozen.items():
        _expect(payload, dotted, expected)

    _exact_keys(_at(payload, "arms"), ARMS, "arms")
    expected_trainable = {
        "B0": [],
        "B1": ["contrastive_projection_head"],
        "B2": [
            "encoder.features.3",
            "response_projection",
            "contrastive_projection_head",
            "training_only_pcr_heads",
        ],
        "B3": [
            "encoder",
            "response_projection",
            "contrastive_projection_head",
            "training_only_pcr_heads",
        ],
    }
    for arm in ARMS:
        _expect(payload, f"arms.{arm}.trainable", expected_trainable[arm])
    _expect(payload, "arms.B0.pcr_training", False)
    for arm in ("B1", "B2", "B3"):
        _expect(payload, f"arms.{arm}.pcr_training", True)
    _expect(payload, "arms.B1.bce_weight", 0.0)
    _expect(payload, "arms.B2.bce_weight", 0.25)
    _expect(payload, "arms.B3.bce_weight", 0.25)

    paths = _at(payload, "paths")
    _exact_keys(
        paths,
        (
            "private_input_repo_root_env",
            "private_input_repo_root_default",
            "stage_b_data_contract",
            "stage_b_data_contract_sha256",
            "clinical_labels",
            "clinical_labels_sha256",
            "fold_manifest",
            "fold_manifest_sha256",
            "ftv_table",
            "ftv_table_sha256",
            "local_confirmation_root",
            "local_checkpoint_pattern",
            "local_feature_pattern",
            "local_feature_metadata_pattern",
        ),
        "paths",
    )
    _expect(payload, "paths.stage_b_data_contract_sha256", LOCKED_STAGE_B_DATA_CONTRACT_SHA256)
    _expect(payload, "paths.clinical_labels_sha256", LOCKED_CLINICAL_SHA256)
    _expect(payload, "paths.fold_manifest_sha256", LOCKED_FOLD_SHA256)
    _expect(payload, "paths.ftv_table_sha256", LOCKED_FTV_SHA256)
    for key in (
        "stage_b_data_contract_sha256",
        "clinical_labels_sha256",
        "fold_manifest_sha256",
        "ftv_table_sha256",
    ):
        require_sha256(paths[key], key)
    if canonical_sha256(payload) != LOCKED_CONFIG_CANONICAL_SHA256:
        raise ValueError("experiment configuration semantic digest drifted")
    return json.loads(json.dumps(payload))


def load_config(
    path: str | Path = CONFIG_PATH,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a config with canonical-path hashing and strict semantic checks."""

    source = Path(path).expanduser().resolve(strict=True)
    if source == CONFIG_PATH.resolve():
        expected_sha256 = LOCKED_CONFIG_SHA256
    if expected_sha256 is not None:
        _verified_file(source, expected_sha256, "experiment configuration")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("experiment configuration is not valid UTF-8 JSON") from exc
    return validate_config(payload)


# More descriptive compatibility name for callers.
load_experiment_config = load_config


@dataclass(frozen=True)
class InputPaths:
    """Hash-verified private inputs resolved from one authoritative root."""

    private_repo_root: Path
    stage_b_data_contract: Path
    clinical_labels: Path
    fold_manifest: Path
    ftv_table: Path
    c1b_cache_manifest: Path
    c1b_cache_manifest_sha256: str
    confirmation_root: Path
    local_checkpoint_pattern: str
    local_feature_pattern: str
    local_feature_metadata_pattern: str

    def checkpoint_path(self, seed: int, fold: int) -> Path:
        seed_value, fold_value = validate_seed_fold(seed, fold)
        relative = self.local_checkpoint_pattern.format(
            seed=seed_value, fold=fold_value
        )
        return _resolve_under(self.confirmation_root, relative, "LOCAL3 checkpoint")

    def feature_path(self, seed: int, fold: int) -> Path:
        seed_value, fold_value = validate_seed_fold(seed, fold)
        relative = self.local_feature_pattern.format(seed=seed_value, fold=fold_value)
        return _resolve_under(self.confirmation_root, relative, "LOCAL3 feature")


def _resolve_under(root: Path, value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its authoritative root") from exc
    return target


def _resolve_configured_path(
    private_root: Path, value: str | Path, label: str
) -> Path:
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else _resolve_under(private_root, raw, label)


def validate_seed_fold(seed: int, fold: int) -> tuple[int, int]:
    if isinstance(seed, bool) or int(seed) not in (*SEEDS, OPTIONAL_SEED):
        raise ValueError(f"seed must be one of {(*SEEDS, OPTIONAL_SEED)}")
    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ValueError("fold must be one of 0..4")
    return int(seed), int(fold)


def _load_stage_b_contract(path: Path) -> MutableMapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-B data contract is not valid UTF-8 JSON") from exc
    fields = (
        "schema_version",
        "fold_manifest",
        "fold_manifest_sha256",
        "technical_eligibility_manifest",
        "technical_eligibility_manifest_sha256",
        "train_only_candidate_manifest",
        "train_only_candidate_manifest_sha256",
        "legacy_cache_manifest",
        "legacy_cache_manifest_sha256",
        "c1b_cache_manifest",
        "c1b_cache_manifest_sha256",
        "ftv_transition_table",
        "ftv_transition_table_sha256",
        "observability_manifest",
        "observability_manifest_sha256",
    )
    _exact_keys(payload, fields, "Stage-B data contract")
    if payload["schema_version"] != 2:
        raise ValueError("Stage-B data contract must have schema_version=2")
    for key, value in payload.items():
        if key.endswith("_sha256"):
            require_sha256(value, key)
    return payload


def resolve_input_paths(
    config: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    verify_hashes: bool = True,
) -> InputPaths:
    """Resolve all inputs and reject redirects away from the private root.

    The historical ``local_confirmation_root`` field pointed at a disposable
    worktree.  The confirmed artifact is therefore deliberately resolved by
    name beneath ``private_repo_root``.  This prevents a stale or substituted
    sibling worktree from becoming the checkpoint authority.
    """

    payload = load_config() if config is None else validate_config(config)
    path_config = payload["paths"]
    environment = os.environ if environ is None else environ
    root_variable = str(path_config["private_input_repo_root_env"])
    root_value = environment.get(
        root_variable, str(path_config["private_input_repo_root_default"])
    )
    private_root = Path(root_value).expanduser().resolve(strict=True)
    if not private_root.is_dir():
        raise ValueError("private input repository root is not a directory")

    stage_b = _resolve_configured_path(
        private_root, path_config["stage_b_data_contract"], "Stage-B data contract"
    )
    clinical = _resolve_configured_path(
        private_root, path_config["clinical_labels"], "clinical labels"
    )
    fold = _resolve_configured_path(
        private_root, path_config["fold_manifest"], "fold manifest"
    )
    ftv = _resolve_configured_path(private_root, path_config["ftv_table"], "FTV table")
    if verify_hashes:
        _verified_file(stage_b, path_config["stage_b_data_contract_sha256"], "Stage-B data contract")
        _verified_file(clinical, path_config["clinical_labels_sha256"], "clinical labels")
        _verified_file(fold, path_config["fold_manifest_sha256"], "fold manifest")
        _verified_file(ftv, path_config["ftv_table_sha256"], "FTV table")
    else:
        for value, label in (
            (stage_b, "Stage-B data contract"),
            (clinical, "clinical labels"),
            (fold, "fold manifest"),
            (ftv, "FTV table"),
        ):
            if not value.is_file():
                raise FileNotFoundError(f"{label} is missing")

    stage_payload = _load_stage_b_contract(stage_b)
    stage_fold = Path(str(stage_payload["fold_manifest"])).expanduser().resolve()
    stage_ftv = Path(str(stage_payload["ftv_transition_table"])).expanduser().resolve()
    if stage_fold != fold or stage_payload["fold_manifest_sha256"] != path_config["fold_manifest_sha256"]:
        raise ValueError("Stage-B and ceiling fold-manifest contracts disagree")
    if stage_ftv != ftv or stage_payload["ftv_transition_table_sha256"] != path_config["ftv_table_sha256"]:
        raise ValueError("Stage-B and ceiling FTV contracts disagree")

    cache_manifest = Path(str(stage_payload["c1b_cache_manifest"])).expanduser().resolve()
    cache_digest = require_sha256(
        stage_payload["c1b_cache_manifest_sha256"], "C1B cache manifest"
    )
    if verify_hashes:
        _verified_file(cache_manifest, cache_digest, "C1B cache manifest")
    elif not cache_manifest.is_file():
        raise FileNotFoundError("C1B cache manifest is missing")

    confirmation = (
        private_root / "additional_experiments" / CONFIRMATION_DIRECTORY
    ).resolve()
    try:
        confirmation.relative_to(private_root)
    except ValueError as exc:  # pragma: no cover - construction makes this defensive.
        raise ValueError("confirmation root escaped private input repository") from exc
    if not confirmation.is_dir():
        raise FileNotFoundError("confirmed LOCAL source root is missing")

    return InputPaths(
        private_repo_root=private_root,
        stage_b_data_contract=stage_b,
        clinical_labels=clinical,
        fold_manifest=fold,
        ftv_table=ftv,
        c1b_cache_manifest=cache_manifest,
        c1b_cache_manifest_sha256=cache_digest,
        confirmation_root=confirmation,
        local_checkpoint_pattern=str(path_config["local_checkpoint_pattern"]),
        local_feature_pattern=str(path_config["local_feature_pattern"]),
        local_feature_metadata_pattern=str(
            path_config["local_feature_metadata_pattern"]
        ),
    )


@dataclass(frozen=True)
class AlignedFullCohort:
    """Private, exactly aligned full-cohort tables."""

    patient_ids: tuple[str, ...]
    clinical: pd.DataFrame
    folds: pd.DataFrame
    cache: pd.DataFrame
    provenance: Mapping[str, Any]

    def split_patient_ids(self, fold: int, split: str) -> tuple[str, ...]:
        _, fold_value = validate_seed_fold(SEEDS[0], fold)
        split_value = str(split).lower()
        if split_value not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        rows = self.folds.loc[
            self.folds["fold"].eq(fold_value)
            & self.folds["split"].eq(split_value),
            "patient_id",
        ]
        return tuple(rows.astype(str))


def _binary(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{label} must contain complete binary values")
    if any(float(value) != int(value) for value in numeric):
        raise ValueError(f"{label} must contain exact binary integers")
    return numeric.astype("int8")


def align_full_cohort_frames(
    clinical: pd.DataFrame,
    folds: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    expected_count: int = FULL_COHORT_SIZE,
    cache_manifest_path: str | Path | None = None,
    verify_cache_files: bool = True,
) -> AlignedFullCohort:
    """Prove exact clinical/fold/cache identity and outcome alignment."""

    if tuple(clinical.columns) != CLINICAL_COLUMNS:
        raise ValueError("clinical adapter schema/order drifted")
    if tuple(folds.columns) != FOLD_COLUMNS:
        raise ValueError("fold adapter schema/order drifted")
    if tuple(cache.columns) != CACHE_COLUMNS:
        raise ValueError("cache adapter schema/order drifted")
    if isinstance(expected_count, bool) or int(expected_count) <= 0:
        raise ValueError("expected cohort count must be a positive integer")
    expected = int(expected_count)

    clinical = clinical.copy()
    folds = folds.copy()
    cache = cache.copy()
    for frame, label in ((clinical, "clinical"), (folds, "fold"), (cache, "cache")):
        frame["patient_id"] = frame["patient_id"].astype(str)
        if frame["patient_id"].eq("").any():
            raise ValueError(f"{label} patient identifiers must be nonempty")
    if len(clinical) != expected or clinical["patient_id"].duplicated().any():
        raise ValueError("clinical table must uniquely contain the frozen full cohort")
    clinical["label_pcr"] = _binary(clinical["label_pcr"], "clinical pCR")
    clinical["label_hr"] = _binary(clinical["label_hr"], "clinical HR")
    clinical["label_her2"] = _binary(clinical["label_her2"], "clinical HER2")
    clinical["arm"] = clinical["arm"].astype("string")
    if clinical["arm"].isna().any() or clinical["arm"].str.len().eq(0).any():
        raise ValueError("assigned treatment arm must be complete and nonempty")

    fold_numeric = pd.to_numeric(folds["fold"], errors="raise")
    if fold_numeric.isna().any() or any(float(value) != int(value) for value in fold_numeric):
        raise ValueError("fold values must be exact integers")
    folds["fold"] = fold_numeric.astype(int)
    folds["split"] = folds["split"].astype(str).str.lower()
    folds["label_pcr"] = _binary(folds["label_pcr"], "fold pCR")
    if set(folds["fold"]) != set(FOLDS) or not set(folds["split"]).issubset(
        {"train", "val", "test"}
    ):
        raise ValueError("fold manifest has an unknown fold or split")
    cohort_ids = tuple(clinical["patient_id"])
    cohort_set = set(cohort_ids)
    for fold in FOLDS:
        current = folds.loc[folds["fold"].eq(fold)]
        if len(current) != expected or current["patient_id"].duplicated().any():
            raise ValueError("every outer fold must uniquely cover the full cohort")
        if set(current["patient_id"]) != cohort_set:
            raise ValueError("clinical and fold patient populations do not align")
        if set(current["split"]) != {"train", "val", "test"}:
            raise ValueError("every outer fold must contain train, val, and test")
    test_counts = (
        folds.assign(_test=folds["split"].eq("test"))
        .groupby("patient_id", sort=False)["_test"]
        .sum()
    )
    if not test_counts.eq(1).all():
        raise ValueError("every full-cohort patient must be outer test exactly once")
    clinical_label = clinical.set_index("patient_id")["label_pcr"]
    aligned_label = folds["patient_id"].map(clinical_label)
    if aligned_label.isna().any() or not aligned_label.astype("int8").eq(
        folds["label_pcr"]
    ).all():
        raise ValueError("clinical and fold pCR labels disagree")

    if cache["patient_id"].duplicated().any() or cache["cache_path"].astype(str).duplicated().any():
        raise ValueError("C1B cache manifest repeats a patient or path")
    if set(cache["input_kind"].astype(str).str.lower()) != {"c1b"}:
        raise ValueError("ceiling experiment accepts only C1B cache entries")
    if not cohort_set.issubset(set(cache["patient_id"])):
        raise ValueError("C1B cache does not cover the full clinical cohort")
    filtered_cache = cache.set_index("patient_id", verify_integrity=True).loc[list(cohort_ids)].reset_index()
    for column in ("cache_size_bytes", "cache_mtime_ns"):
        numeric = pd.to_numeric(filtered_cache[column], errors="raise")
        if numeric.isna().any() or not pd.api.types.is_integer_dtype(numeric.dtype):
            raise ValueError(f"cache {column} must contain exact integers")
        filtered_cache[column] = numeric.astype("int64")
    if filtered_cache["cache_size_bytes"].le(0).any() or filtered_cache["cache_mtime_ns"].lt(0).any():
        raise ValueError("cache size/mtime pins are invalid")
    for digest in filtered_cache["cache_sha256"]:
        require_sha256(digest, "cache content")

    manifest_parent = (
        Path(cache_manifest_path).expanduser().resolve().parent
        if cache_manifest_path is not None
        else None
    )
    if verify_cache_files:
        for row in filtered_cache.itertuples(index=False):
            raw = Path(str(row.cache_path)).expanduser()
            path = raw.resolve() if raw.is_absolute() else (
                (manifest_parent / raw).resolve() if manifest_parent is not None else raw.resolve()
            )
            observed = path.stat()
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError("a pinned C1B cache path is not a regular file")
            if observed.st_size != int(row.cache_size_bytes) or observed.st_mtime_ns != int(row.cache_mtime_ns):
                raise ValueError("a pinned C1B cache stat changed")

    order = {patient_id: index for index, patient_id in enumerate(cohort_ids)}
    folds["_order"] = folds["patient_id"].map(order)
    folds = folds.sort_values(["fold", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)
    provenance = {
        "population": FULL_COHORT_NAME if expected == FULL_COHORT_SIZE else "test_fixture",
        "patient_count": expected,
        "fold_count": len(FOLDS),
        "fold_row_count": len(folds),
        "cache_patient_count": len(filtered_cache),
        "clinical_fold_labels_exact": True,
        "cache_files_stat_verified": bool(verify_cache_files),
    }
    return AlignedFullCohort(
        patient_ids=cohort_ids,
        clinical=clinical.reset_index(drop=True),
        folds=folds,
        cache=filtered_cache,
        provenance=provenance,
    )


def load_aligned_full_cohort(
    config: Mapping[str, Any] | None = None,
    paths: InputPaths | None = None,
    *,
    verify_cache_files: bool = True,
) -> AlignedFullCohort:
    """Load the hash-pinned full-808 clinical/fold/cache intersection."""

    payload = load_config() if config is None else validate_config(config)
    resolved = resolve_input_paths(payload) if paths is None else paths
    clinical = pd.read_csv(resolved.clinical_labels, usecols=list(CLINICAL_COLUMNS))
    folds = pd.read_csv(resolved.fold_manifest, usecols=list(FOLD_COLUMNS))
    cache = pd.read_csv(resolved.c1b_cache_manifest, usecols=list(CACHE_COLUMNS))
    return align_full_cohort_frames(
        clinical,
        folds,
        cache,
        expected_count=int(payload["populations"][FULL_COHORT_NAME]),
        cache_manifest_path=resolved.c1b_cache_manifest,
        verify_cache_files=verify_cache_files,
    )


# Compatibility aliases used by orchestration code and focused contract tests.
load_full_808 = load_aligned_full_cohort
load_full_808_alignment = load_aligned_full_cohort


__all__ = [
    "ARMS",
    "AlignedFullCohort",
    "CACHE_COLUMNS",
    "CLINICAL_COLUMNS",
    "CONFIG_PATH",
    "FOLDS",
    "FOLD_COLUMNS",
    "FULL_COHORT_NAME",
    "FULL_COHORT_SIZE",
    "InputPaths",
    "MATCHING_FIELDS",
    "OPTIONAL_SEED",
    "PRIMARY_TIMINGS",
    "SEEDS",
    "SUPPLEMENTARY_TIMING",
    "TIMINGS",
    "align_full_cohort_frames",
    "canonical_sha256",
    "file_sha256",
    "load_aligned_full_cohort",
    "load_config",
    "load_experiment_config",
    "load_full_808",
    "load_full_808_alignment",
    "require_sha256",
    "resolve_input_paths",
    "validate_config",
    "validate_seed_fold",
]
