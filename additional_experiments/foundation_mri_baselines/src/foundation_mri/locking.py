"""Hash-locked formal evaluation inputs and executable protocol files.

The model/input extraction lock is intentionally separate from this lock.  A
formal outcome run must pass both the loader-level schema/provenance checks and
this exact evaluation manifest.  Synthetic unit tests may bypass the manifest
only through their explicit ``allow_unlocked_inputs`` path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_V2_CONSUMERS = ("baseline", "probe", "summarizer")
_V2_RUN_CONSUMERS = ("baseline", "probe")
_V3_CONSUMERS = ("probe", "summarizer")
_V3_RUN_CONSUMERS = ("probe",)
_FORMAL_FOUNDATION_MODELS = (
    "medicalnet_resnet50_3dseg8",
    "dino_vitb16_imagenet1k",
)
_FINALIZATION_LOCK_SHA256 = (
    "30c0c0e6ce7d92fb6164368addf467e2142e5159fb95240f4c030ddb986c4e7b"
)
_REPORTING_LOCK_SCHEMA = "foundation_mri_reporting_lock_v1"
_EMPTY_ARGUMENT_VECTOR_SHA256 = (
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
)


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value with one unambiguous, whitespace-free encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def argument_vector_sha256(arguments: Sequence[str | Path]) -> str:
    """Length-prefix an argv vector so shell quoting/joining cannot alter identity.

    The vector excludes the Python executable and script name.  Each caller must
    pass the exact sequence handed to its argument parser.
    """

    values = tuple(str(value) for value in arguments)
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_digest(value: object, label: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _checked_lower_digest(value: object, label: str) -> str:
    text = str(value).strip()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _root_relative_file(root: Path, raw_path: object, label: str) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError(f"{label} path is empty")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be root-relative without traversal")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escaped experiment root") from error
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    return path


def _verify_root_record(root: Path, record: Mapping[str, object], label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _root_relative_file(root, record["path"], label)
    expected = _checked_lower_digest(record["sha256"], f"{label} SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 drifted: expected={expected}, observed={observed}"
        )
    return path


def _path_within_root(
    root: Path, raw_path: str | Path, label: str, *, must_exist: bool
) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay within experiment root") from error
    if must_exist and not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return resolved


def _resolve(root: Path, raw_path: object, label: str) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError(f"{label} path is empty")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return resolved


def _verify_record(root: Path, record: Mapping[str, object], label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(root, record["path"], label)
    expected = _checked_digest(record["sha256"], f"{label} SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 drifted: expected={expected}, observed={observed}"
        )
    return path


def _normalise_paths(values: Iterable[str | Path]) -> tuple[Path, ...]:
    return tuple(
        sorted((Path(value).resolve(strict=True) for value in values), key=str)
    )


def _load_and_verify_code_lock(
    experiment_root: str | Path, lock_path: str | Path
) -> tuple[Path, Path, dict[str, object]]:
    root = Path(experiment_root).resolve(strict=True)
    source = Path(lock_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "lock_kind",
        "locked_at",
        "locked_before_formal_test_evaluation",
        "formal_test_outcomes_seen",
        "files",
        "inputs",
        "execution",
    }
    if set(payload) != required:
        raise ValueError("evaluation lock top-level schema drifted")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported evaluation lock schema")
    if payload["lock_kind"] != "formal_outcome_evaluation":
        raise ValueError("wrong evaluation lock kind")
    if payload["locked_before_formal_test_evaluation"] is not True:
        raise ValueError("evaluation lock was not declared pre-test")
    if payload["formal_test_outcomes_seen"] is not False:
        raise ValueError(
            "evaluation lock claims formal test outcomes were already seen"
        )

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("evaluation lock has no executable/config file hashes")
    for relative, digest in sorted(files.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                "evaluation code lock paths must stay within experiment root"
            )
        path = (root / relative_path).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("evaluation code lock escaped experiment root") from error
        expected = _checked_digest(digest, f"locked file {relative}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"evaluation file drifted: {relative}; expected={expected}, observed={observed}"
            )
    return root, source, payload


def _read_lock_payload(
    experiment_root: str | Path, lock_path: str | Path
) -> tuple[Path, Path, dict[str, object]]:
    root = Path(experiment_root).resolve(strict=True)
    source = Path(lock_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("evaluation lock is not a regular file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation lock must contain a JSON object")
    return root, source, payload


def _validate_v1_record_shape(record: object, label: str) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    if not str(record["path"]).strip():
        raise ValueError(f"{label} path is empty")
    _checked_digest(record["sha256"], f"{label} SHA-256")


def _validate_v1_input_schema(inputs: object) -> Mapping[str, object]:
    if not isinstance(inputs, dict) or set(inputs) != {
        "fold_manifest",
        "clinical_labels",
        "radiomics",
        "foundation_features",
        "current_cnn_features",
    }:
        raise ValueError("evaluation input lock schema drifted")
    for key in ("fold_manifest", "clinical_labels", "radiomics"):
        _validate_v1_record_shape(inputs[key], key.replace("_", " "))

    foundation = inputs["foundation_features"]
    if not isinstance(foundation, list) or not foundation:
        raise ValueError("evaluation lock has no foundation features")
    models: set[str] = set()
    for record in foundation:
        if not isinstance(record, dict) or set(record) != {"model", "path", "sha256"}:
            raise ValueError("foundation feature lock record schema drifted")
        model = str(record["model"]).strip()
        if not model or model in models:
            raise ValueError("foundation feature model identity is empty/duplicated")
        models.add(model)
        _validate_v1_record_shape(
            {"path": record["path"], "sha256": record["sha256"]},
            f"foundation feature {model}",
        )

    cnn = inputs["current_cnn_features"]
    if not isinstance(cnn, list) or not cnn:
        raise ValueError("evaluation lock has no current-CNN features")
    folds: dict[tuple[str, str], set[int]] = {}
    for record in cnn:
        if not isinstance(record, dict) or set(record) != {
            "model",
            "spatial",
            "fold",
            "feature",
            "metadata",
        }:
            raise ValueError("current-CNN feature lock record schema drifted")
        model = str(record["model"]).strip()
        spatial = str(record["spatial"]).strip().upper()
        try:
            fold = int(record["fold"])
        except (TypeError, ValueError) as error:
            raise ValueError("current-CNN lock fold is invalid/duplicated") from error
        key = (model, spatial)
        if key not in {("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL")}:
            raise ValueError("evaluation lock contains an unapproved current-CNN arm")
        if fold not in range(5) or fold in folds.setdefault(key, set()):
            raise ValueError("current-CNN lock fold is invalid/duplicated")
        folds[key].add(fold)
        _validate_v1_record_shape(record["feature"], "current-CNN feature")
        _validate_v1_record_shape(record["metadata"], "current-CNN metadata")
    if set(folds) != {("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL")} or any(
        values != set(range(5)) for values in folds.values()
    ):
        raise ValueError("evaluation lock must contain exactly five GAP0/LOCAL0 folds")
    return inputs


def _validate_v1_manifest_only(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "lock_kind",
        "locked_at",
        "locked_before_formal_test_evaluation",
        "formal_test_outcomes_seen",
        "files",
        "inputs",
        "execution",
    }
    if set(payload) != required:
        raise ValueError("parent evaluation lock top-level schema drifted")
    if payload["schema_version"] != 1:
        raise ValueError("chained parent must use evaluation lock schema 1")
    if payload["lock_kind"] != "formal_outcome_evaluation":
        raise ValueError("wrong parent evaluation lock kind")
    if payload["locked_before_formal_test_evaluation"] is not True:
        raise ValueError("parent evaluation lock was not declared pre-test")
    if payload["formal_test_outcomes_seen"] is not False:
        raise ValueError("parent lock claims formal test outcomes were already seen")
    if not str(payload["locked_at"]).strip():
        raise ValueError("parent evaluation lock timestamp is empty")
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("parent evaluation lock has no file hashes")
    for relative, digest in files.items():
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("parent code lock paths must stay within experiment root")
        _checked_digest(digest, f"parent locked file {relative}")
    _validate_v1_input_schema(payload["inputs"])
    if not isinstance(payload["execution"], dict):
        raise ValueError("parent execution lock must be an object")


def _load_and_verify_v2_chain(
    experiment_root: str | Path,
    lock_path: str | Path,
    *,
    verify_final_files: bool = True,
) -> tuple[Path, Path, dict[str, object], Path, dict[str, object]]:
    root, source, payload = _read_lock_payload(experiment_root, lock_path)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "schema-v2 evaluation lock must stay within experiment root"
        ) from error

    required = {
        "schema_version",
        "lock_kind",
        "lock_generation",
        "locked_at",
        "locked_before_formal_test_evaluation",
        "formal_test_outcomes_seen",
        "parent_lock",
        "aborted_v1_attempts",
        "prelock_state",
        "inherited_inputs",
        "development_provenance",
        "files",
        "execution",
    }
    if set(payload) != required:
        raise ValueError("schema-v2 evaluation lock top-level schema drifted")
    if payload["schema_version"] != 2:
        raise ValueError("unsupported chained evaluation lock schema")
    if payload["lock_kind"] != "formal_outcome_evaluation_chain":
        raise ValueError("wrong chained evaluation lock kind")
    if payload["lock_generation"] != "whole_evaluation_v2":
        raise ValueError("wrong chained evaluation lock generation")
    if not str(payload["locked_at"]).strip():
        raise ValueError("schema-v2 evaluation lock timestamp is empty")
    if payload["locked_before_formal_test_evaluation"] is not True:
        raise ValueError("schema-v2 evaluation lock was not declared pre-test")
    if payload["formal_test_outcomes_seen"] is not False:
        raise ValueError("schema-v2 lock claims formal test outcomes were already seen")

    attempts = payload["aborted_v1_attempts"]
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise ValueError("schema-v2 lock must record exactly two aborted v1 attempts")
    expected_attempts = ("baseline", "probe")
    for record, consumer in zip(attempts, expected_attempts, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "consumer",
            "exit_code",
            "outputs_written",
            "metric_values_viewed",
        }:
            raise ValueError("aborted v1 attempt schema drifted")
        if (
            record["consumer"] != consumer
            or record["exit_code"] != 143
            or record["outputs_written"] is not False
            or record["metric_values_viewed"] is not False
        ):
            raise ValueError("aborted v1 attempt provenance is inconsistent")

    state = payload["prelock_state"]
    if not isinstance(state, dict) or set(state) != {
        "baseline_outputs_exist",
        "probe_outputs_exist",
        "metric_values_viewed",
    }:
        raise ValueError("schema-v2 prelock state schema drifted")
    if any(value is not False for value in state.values()):
        raise ValueError("schema-v2 prelock state must be metric-free with no outputs")

    parent_source = _verify_root_record(root, payload["parent_lock"], "parent lock")
    parent_payload = json.loads(parent_source.read_text(encoding="utf-8"))
    if not isinstance(parent_payload, dict):
        raise ValueError("parent evaluation lock must contain a JSON object")
    _validate_v1_manifest_only(parent_payload)

    inherited = payload["inherited_inputs"]
    if not isinstance(inherited, dict) or set(inherited) != {"source", "sha256"}:
        raise ValueError("schema-v2 inherited-input contract drifted")
    if inherited["source"] != "parent_lock.inputs":
        raise ValueError("schema-v2 inputs must be inherited from parent_lock.inputs")
    expected_inputs = _checked_lower_digest(
        inherited["sha256"], "inherited inputs SHA-256"
    )
    observed_inputs = canonical_json_sha256(parent_payload["inputs"])
    if expected_inputs != observed_inputs:
        raise ValueError("parent input contract SHA-256 drifted")

    development = payload["development_provenance"]
    if not isinstance(development, dict) or set(development) != {
        "source_snapshot",
        "patch_manifest",
        "smoke_receipt",
    }:
        raise ValueError("schema-v2 development provenance schema drifted")
    for name in ("source_snapshot", "patch_manifest", "smoke_receipt"):
        _verify_root_record(root, development[name], name.replace("_", " "))

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("schema-v2 evaluation lock has no final file hashes")
    for relative, digest in sorted(files.items()):
        expected = _checked_lower_digest(
            digest, f"final locked file {relative} SHA-256"
        )
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("schema-v2 locked file path escaped experiment root")
        if verify_final_files:
            path = _root_relative_file(root, relative, f"final locked file {relative}")
            observed = file_sha256(path)
            if observed != expected:
                raise ValueError(
                    f"evaluation file drifted: {relative}; expected={expected}, observed={observed}"
                )

    execution = payload["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "parent_execution_sha256",
        "consumer_argv",
    }:
        raise ValueError("schema-v2 execution contract drifted")
    expected_execution = _checked_lower_digest(
        execution["parent_execution_sha256"], "parent execution SHA-256"
    )
    if expected_execution != canonical_json_sha256(parent_payload["execution"]):
        raise ValueError("parent execution contract SHA-256 drifted")
    consumers = execution["consumer_argv"]
    if not isinstance(consumers, dict) or set(consumers) != set(_V2_CONSUMERS):
        raise ValueError("schema-v2 lock must define baseline/probe/summarizer argv")
    for consumer, record in consumers.items():
        if not isinstance(record, dict) or set(record) != {"argc", "sha256"}:
            raise ValueError(f"{consumer} argv contract schema drifted")
        if isinstance(record["argc"], bool) or not isinstance(record["argc"], int):
            raise ValueError(f"{consumer} argv count must be a nonnegative integer")
        if record["argc"] < 0:
            raise ValueError(f"{consumer} argv count must be a nonnegative integer")
        _checked_lower_digest(record["sha256"], f"{consumer} argv SHA-256")
    return root, source, payload, parent_source, parent_payload


def _verify_v2_consumer(
    payload: Mapping[str, object],
    expected_consumer: str,
    command_argv: Sequence[str | Path] | None,
) -> tuple[int, str]:
    if expected_consumer not in _V2_CONSUMERS:
        raise ValueError("expected_consumer must be baseline, probe, or summarizer")
    if command_argv is None:
        raise ValueError("schema-v2 verification requires the exact command argv")
    values = tuple(str(value) for value in command_argv)
    contract = payload["execution"]["consumer_argv"][expected_consumer]  # type: ignore[index]
    observed = argument_vector_sha256(values)
    if contract["argc"] != len(values) or contract["sha256"] != observed:
        raise ValueError(f"{expected_consumer} command argv does not match the lock")
    return len(values), observed


def _v2_code_receipt(
    root: Path,
    source: Path,
    payload: Mapping[str, object],
    parent_source: Path,
    *,
    expected_consumer: str,
    argument_count: int,
    argument_sha256: str,
) -> dict[str, object]:
    development = payload["development_provenance"]
    return {
        "schema_version": 2,
        "lock_generation": "whole_evaluation_v2",
        "lock_path": source.relative_to(root).as_posix(),
        "lock_sha256": file_sha256(source),
        "parent_lock_sha256": file_sha256(parent_source),
        "inherited_inputs_sha256": payload["inherited_inputs"]["sha256"],  # type: ignore[index]
        "verified_file_count": len(payload["files"]),
        "expected_consumer": expected_consumer,
        "argument_count": argument_count,
        "argument_vector_sha256": argument_sha256,
        "source_snapshot_sha256": development["source_snapshot"]["sha256"],  # type: ignore[index]
        "patch_manifest_sha256": development["patch_manifest"]["sha256"],  # type: ignore[index]
        "smoke_receipt_sha256": development["smoke_receipt"]["sha256"],  # type: ignore[index]
        "execution": payload["execution"],
    }


def _verify_v2_to_v3_source_snapshot(
    *,
    root: Path,
    snapshot_source: Path,
    patch_source: Path,
    v2_source: Path,
    v2_payload: Mapping[str, object],
    v3_payload: Mapping[str, object],
) -> None:
    snapshot = json.loads(snapshot_source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "snapshot_kind",
        "v2_lock",
        "forward_patch",
        "changed_v2_locked_files",
        "reconstruction",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("v2-to-v3 source snapshot schema drifted")
    if (
        snapshot["schema_version"] != 1
        or snapshot["snapshot_kind"]
        != "reconstructable_evaluation_v2_to_v3_source_transition"
    ):
        raise ValueError("wrong v2-to-v3 source snapshot identity")
    lock_record = snapshot["v2_lock"]
    if not isinstance(lock_record, dict) or set(lock_record) != {
        "path",
        "sha256",
        "locked_file_count",
    }:
        raise ValueError("v2-to-v3 snapshot v2-lock schema drifted")
    if (
        _root_relative_file(root, lock_record["path"], "snapshot v2 lock")
        != v2_source
        or _checked_lower_digest(lock_record["sha256"], "snapshot v2 lock SHA-256")
        != file_sha256(v2_source)
        or lock_record["locked_file_count"] != len(v2_payload["files"])  # type: ignore[arg-type]
    ):
        raise ValueError("v2-to-v3 snapshot parent-lock identity drifted")
    patch_record = snapshot["forward_patch"]
    if not isinstance(patch_record, dict) or set(patch_record) != {
        "path",
        "sha256",
        "format",
        "line_count",
        "byte_count",
    }:
        raise ValueError("v2-to-v3 forward-patch schema drifted")
    if (
        _root_relative_file(root, patch_record["path"], "snapshot forward patch")
        != patch_source
        or _checked_lower_digest(
            patch_record["sha256"], "snapshot forward patch SHA-256"
        )
        != file_sha256(patch_source)
        or patch_record["format"] != "unified_diff_a_to_b_strip_1"
        or patch_record["byte_count"] != patch_source.stat().st_size
        or patch_record["line_count"]
        != len(patch_source.read_bytes().splitlines())
    ):
        raise ValueError("v2-to-v3 forward-patch identity drifted")
    v2_files = v2_payload["files"]
    v3_files = v3_payload["files"]
    if not isinstance(v2_files, dict) or not isinstance(v3_files, dict):
        raise ValueError("v2-to-v3 locked file maps are invalid")
    if not set(v2_files).issubset(v3_files):
        raise ValueError("v3 final files do not retain every v2 locked path")
    expected_changed = {
        relative
        for relative, digest in v2_files.items()
        if v3_files[relative] != digest
    }
    changed = snapshot["changed_v2_locked_files"]
    if not isinstance(changed, dict) or set(changed) != expected_changed:
        raise ValueError("v2-to-v3 changed-file manifest drifted")
    for relative, record in changed.items():
        if not isinstance(record, dict) or set(record) != {"v2_sha256", "v3_sha256"}:
            raise ValueError("v2-to-v3 changed-file record schema drifted")
        if (
            record["v2_sha256"] != v2_files[relative]
            or record["v3_sha256"] != v3_files[relative]
        ):
            raise ValueError(f"v2-to-v3 changed-file hashes drifted: {relative}")
    reconstruction = snapshot["reconstruction"]
    if not isinstance(reconstruction, dict) or set(reconstruction) != {
        "v2_from_v3_command",
        "v3_from_v2_command",
        "verify_v2_command",
        "forward_patch_verified",
        "reverse_patch_verified",
    }:
        raise ValueError("v2-to-v3 reconstruction schema drifted")
    if (
        reconstruction["forward_patch_verified"] is not True
        or reconstruction["reverse_patch_verified"] is not True
        or not all(
            str(reconstruction[key]).strip()
            for key in (
                "v2_from_v3_command",
                "v3_from_v2_command",
                "verify_v2_command",
            )
        )
    ):
        raise ValueError("v2-to-v3 reconstruction was not verified")


def _load_and_verify_v3_chain(
    experiment_root: str | Path,
    lock_path: str | Path,
    *,
    verify_final_files: bool = True,
) -> tuple[
    Path,
    Path,
    dict[str, object],
    Path,
    dict[str, object],
    Path,
    dict[str, object],
]:
    """Verify active v3 code plus immutable v2/v1 and mixed-producer lineage."""

    root, source, payload = _read_lock_payload(experiment_root, lock_path)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("schema-v3 evaluation lock must stay within experiment root") from error
    required = {
        "schema_version",
        "lock_kind",
        "lock_generation",
        "locked_at",
        "locked_after_baseline_v2_completion",
        "baseline_v2_metric_prediction_values_seen",
        "locked_before_formal_probe_v3_evaluation",
        "formal_probe_v3_outcomes_seen",
        "parent_lock",
        "finalization_lock",
        "aborted_v2_attempts",
        "prelock_state",
        "inherited_inputs",
        "historical_producers",
        "development_provenance",
        "files",
        "execution",
    }
    if set(payload) != required:
        raise ValueError("schema-v3 evaluation lock top-level schema drifted")
    if (
        payload["schema_version"] != 3
        or payload["lock_kind"] != "formal_outcome_evaluation_chain"
        or payload["lock_generation"] != "probe_retry_v3"
    ):
        raise ValueError("wrong schema-v3 evaluation lock identity")
    if not str(payload["locked_at"]).strip():
        raise ValueError("schema-v3 evaluation lock timestamp is empty")
    if (
        payload["locked_after_baseline_v2_completion"] is not True
        or payload["baseline_v2_metric_prediction_values_seen"] is not False
        or payload["locked_before_formal_probe_v3_evaluation"] is not True
        or payload["formal_probe_v3_outcomes_seen"] is not False
    ):
        raise ValueError("schema-v3 timing/value-visibility declarations are inconsistent")

    v2_source = _verify_root_record(root, payload["parent_lock"], "v3 parent lock")
    (
        _,
        verified_v2_source,
        v2_payload,
        v1_source,
        v1_payload,
    ) = _load_and_verify_v2_chain(root, v2_source, verify_final_files=False)
    if verified_v2_source != v2_source:
        raise AssertionError("schema-v3 parent v2 source drifted")
    finalization_source = _verify_root_record(
        root, payload["finalization_lock"], "finalization lock"
    )
    if file_sha256(finalization_source) != _FINALIZATION_LOCK_SHA256:
        raise ValueError("FINALIZATION_LOCK.v1 SHA-256 drifted")

    attempts = payload["aborted_v2_attempts"]
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("schema-v3 must record exactly one failed v2 probe")
    attempt = attempts[0]
    expected_attempt_keys = {
        "consumer",
        "producer_lock_sha256",
        "exit_code",
        "outputs_written",
        "metric_values_viewed",
        "original_traceback_recovered",
        "formal_exception_claimed",
        "failure_window",
        "synthetic_diagnostic_scope",
        "progress_log",
    }
    if not isinstance(attempt, dict) or set(attempt) != expected_attempt_keys:
        raise ValueError("failed v2 probe record schema drifted")
    if (
        attempt["consumer"] != "probe"
        or attempt["producer_lock_sha256"] != file_sha256(v2_source)
        or attempt["exit_code"] != 1
        or attempt["outputs_written"] is not False
        or attempt["metric_values_viewed"] is not False
        or attempt["original_traceback_recovered"] is not False
        or attempt["formal_exception_claimed"] is not False
        or attempt["failure_window"]
        != "ridge_fold3_after_selection_before_fold4_start"
        or attempt["synthetic_diagnostic_scope"]
        != "matching_failure_path_not_recovered_formal_exception"
    ):
        raise ValueError("failed v2 probe provenance is inconsistent")
    progress = attempt["progress_log"]
    progress_keys = {
        "path",
        "sha256",
        "mode_octal",
        "byte_count",
        "line_count",
        "ends_with_newline",
        "event_counts",
        "last_event",
    }
    if not isinstance(progress, dict) or set(progress) != progress_keys:
        raise ValueError("failed v2 probe progress schema drifted")
    progress_source = _verify_root_record(
        root,
        {"path": progress["path"], "sha256": progress["sha256"]},
        "failed v2 probe progress",
    )
    expected_event_counts = {
        "formal_run_started": 1,
        "fold_started": 69,
        "fold_completed": 69,
        "candidate_started": 540,
        "candidate_completed": 540,
        "estimator_started": 720,
        "estimator_completed": 720,
    }
    if (
        progress["mode_octal"] != "0600"
        or progress["byte_count"] != progress_source.stat().st_size
        or progress["line_count"] != 2659
        or progress["ends_with_newline"] is not True
        or progress["event_counts"] != expected_event_counts
        or progress["last_event"] != "fold_completed"
        or oct(progress_source.stat().st_mode & 0o777) != "0o600"
    ):
        raise ValueError("failed v2 probe progress metadata drifted")

    state = payload["prelock_state"]
    if not isinstance(state, dict) or set(state) != {
        "baseline_v2_completed",
        "baseline_v2_receipt_bound",
        "probe_v3_outputs_exist",
        "metric_values_viewed",
    }:
        raise ValueError("schema-v3 prelock-state schema drifted")
    if state != {
        "baseline_v2_completed": True,
        "baseline_v2_receipt_bound": True,
        "probe_v3_outputs_exist": False,
        "metric_values_viewed": False,
    }:
        raise ValueError("schema-v3 prelock state is inconsistent")
    inherited = payload["inherited_inputs"]
    if not isinstance(inherited, dict) or set(inherited) != {"source", "sha256"}:
        raise ValueError("schema-v3 inherited-input schema drifted")
    if (
        inherited["source"] != "parent_lock.inherited_inputs"
        or inherited["sha256"] != v2_payload["inherited_inputs"]["sha256"]  # type: ignore[index]
    ):
        raise ValueError("schema-v3 inherited input contract drifted")

    historical = payload["historical_producers"]
    if not isinstance(historical, dict) or set(historical) != {"baseline_v2"}:
        raise ValueError("schema-v3 historical producer schema drifted")
    baseline = historical["baseline_v2"]
    if not isinstance(baseline, dict) or set(baseline) != {
        "consumer",
        "producer_lock",
        "run_receipt",
        "argv",
        "artifacts",
        "audit_counts",
    }:
        raise ValueError("historical baseline-v2 schema drifted")
    if baseline["consumer"] != "baseline":
        raise ValueError("historical producer is not baseline-v2")
    producer_lock = _verify_root_record(
        root, baseline["producer_lock"], "historical baseline producer lock"
    )
    if producer_lock != v2_source:
        raise ValueError("historical baseline producer lock differs from v3 parent")
    receipt_source = _verify_root_record(
        root, baseline["run_receipt"], "historical baseline run receipt"
    )
    receipt = json.loads(receipt_source.read_text(encoding="utf-8"))
    receipt_required = {
        "schema_version",
        "receipt_kind",
        "consumer",
        "lock_sha256",
        "parent_lock_sha256",
        "argument_count",
        "argument_vector_sha256",
        "artifact_hash_method",
        "metric_values_viewed",
        "artifacts",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_required:
        raise ValueError("historical baseline receipt schema drifted")
    argv = baseline["argv"]
    if not isinstance(argv, dict) or set(argv) != {"argc", "sha256"}:
        raise ValueError("historical baseline argv schema drifted")
    if (
        receipt["schema_version"] != 1
        or receipt["receipt_kind"] != "formal_metric_free_run_provenance"
        or receipt["consumer"] != "baseline"
        or receipt["lock_sha256"] != file_sha256(v2_source)
        or receipt["parent_lock_sha256"] != file_sha256(v1_source)
        or receipt["artifact_hash_method"] != "sha256_binary_stream_no_parse"
        or receipt["metric_values_viewed"] is not False
        or receipt["argument_count"] != argv["argc"]
        or receipt["argument_vector_sha256"] != argv["sha256"]
        or argv != v2_payload["execution"]["consumer_argv"]["baseline"]  # type: ignore[index]
    ):
        raise ValueError("historical baseline receipt/argv lineage drifted")
    artifacts = baseline["artifacts"]
    role_to_receipt = {
        "predictions": "baseline_predictions_private",
        "selection": "baseline_selection_private",
        "metrics": "baseline_metrics_public",
        "progress": "baseline_progress_private",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(role_to_receipt):
        raise ValueError("historical baseline artifact schema drifted")
    if not isinstance(receipt["artifacts"], dict) or set(receipt["artifacts"]) != set(
        role_to_receipt.values()
    ):
        raise ValueError("historical baseline receipt artifact schema drifted")
    for role, receipt_key in role_to_receipt.items():
        historical_path = _verify_root_record(
            root, artifacts[role], f"historical baseline artifact {role}"
        )
        receipt_record = receipt["artifacts"][receipt_key]
        if (
            not isinstance(receipt_record, dict)
            or set(receipt_record) != {"path", "sha256"}
            or receipt_record != artifacts[role]
            or historical_path
            != _root_relative_file(
                root, receipt_record["path"], f"baseline receipt artifact {role}"
            )
        ):
            raise ValueError(f"historical baseline artifact lineage drifted: {role}")
    audit_counts = baseline["audit_counts"]
    expected_audit_counts = {
        "prediction_rows": 64137,
        "selection_rows": 630,
        "public_metric_rows": 252,
        "progress_fold_completed": 630,
        "progress_candidate_completed": 11340,
        "max_observed_n_iter": 1485,
        "capped_candidate_count": 0,
    }
    if audit_counts != expected_audit_counts:
        raise ValueError("historical baseline-v2 audited counts drifted")

    development = payload["development_provenance"]
    if not isinstance(development, dict) or set(development) != {
        "source_snapshot",
        "patch_manifest",
        "smoke_receipt",
    }:
        raise ValueError("schema-v3 development provenance schema drifted")
    snapshot_source = _verify_root_record(
        root, development["source_snapshot"], "v2-to-v3 source snapshot"
    )
    patch_source = _verify_root_record(
        root, development["patch_manifest"], "v2-to-v3 patch manifest"
    )
    _verify_root_record(root, development["smoke_receipt"], "v3 smoke receipt")

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("schema-v3 evaluation lock has no final file hashes")
    for relative, digest in sorted(files.items()):
        expected = _checked_lower_digest(
            digest, f"v3 final locked file {relative} SHA-256"
        )
        if verify_final_files:
            path = _root_relative_file(root, relative, f"v3 final locked file {relative}")
            observed = file_sha256(path)
            if observed != expected:
                raise ValueError(
                    f"evaluation file drifted: {relative}; expected={expected}, observed={observed}"
                )
    _verify_v2_to_v3_source_snapshot(
        root=root,
        snapshot_source=snapshot_source,
        patch_source=patch_source,
        v2_source=v2_source,
        v2_payload=v2_payload,
        v3_payload=payload,
    )
    execution = payload["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "parent_execution_sha256",
        "consumer_argv",
        "producer_order",
    }:
        raise ValueError("schema-v3 execution contract drifted")
    if execution["parent_execution_sha256"] != canonical_json_sha256(
        v2_payload["execution"]
    ):
        raise ValueError("schema-v3 parent execution contract drifted")
    consumers = execution["consumer_argv"]
    if not isinstance(consumers, dict) or set(consumers) != set(_V3_CONSUMERS):
        raise ValueError("schema-v3 must define only probe/summarizer argv")
    for consumer, record in consumers.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"argc", "sha256"}
            or isinstance(record["argc"], bool)
            or not isinstance(record["argc"], int)
            or record["argc"] < 0
        ):
            raise ValueError(f"schema-v3 {consumer} argv contract drifted")
        _checked_lower_digest(record["sha256"], f"schema-v3 {consumer} argv SHA-256")
    if execution["producer_order"] != ["baseline:v2", "probe:v3", "summarizer:v3"]:
        raise ValueError("schema-v3 producer order drifted")
    return root, source, payload, v2_source, v2_payload, v1_source, v1_payload


def _verify_v3_consumer(
    payload: Mapping[str, object],
    expected_consumer: str,
    command_argv: Sequence[str | Path] | None,
) -> tuple[int, str]:
    if expected_consumer not in _V3_CONSUMERS:
        raise ValueError("schema-v3 expected_consumer must be probe or summarizer")
    if command_argv is None:
        raise ValueError("schema-v3 verification requires the exact command argv")
    values = tuple(str(value) for value in command_argv)
    contract = payload["execution"]["consumer_argv"][expected_consumer]  # type: ignore[index]
    digest = argument_vector_sha256(values)
    if contract["argc"] != len(values) or contract["sha256"] != digest:
        raise ValueError(f"{expected_consumer} command argv does not match the v3 lock")
    return len(values), digest


def _v3_code_receipt(
    root: Path,
    source: Path,
    payload: Mapping[str, object],
    v2_source: Path,
    *,
    expected_consumer: str,
    argument_count: int,
    argument_sha256: str,
) -> dict[str, object]:
    development = payload["development_provenance"]
    return {
        "schema_version": 3,
        "lock_generation": "probe_retry_v3",
        "lock_path": source.relative_to(root).as_posix(),
        "lock_sha256": file_sha256(source),
        "parent_lock_sha256": file_sha256(v2_source),
        "inherited_inputs_sha256": payload["inherited_inputs"]["sha256"],  # type: ignore[index]
        "verified_file_count": len(payload["files"]),
        "expected_consumer": expected_consumer,
        "argument_count": argument_count,
        "argument_vector_sha256": argument_sha256,
        "source_snapshot_sha256": development["source_snapshot"]["sha256"],  # type: ignore[index]
        "patch_manifest_sha256": development["patch_manifest"]["sha256"],  # type: ignore[index]
        "smoke_receipt_sha256": development["smoke_receipt"]["sha256"],  # type: ignore[index]
        "finalization_lock_sha256": payload["finalization_lock"]["sha256"],  # type: ignore[index]
        "execution": payload["execution"],
    }


def _verify_v3_to_reporting_source_snapshot(
    *,
    root: Path,
    snapshot_source: Path,
    patch_source: Path,
    v3_source: Path,
    v3_payload: Mapping[str, object],
    reporting_payload: Mapping[str, object],
) -> None:
    snapshot = json.loads(snapshot_source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "snapshot_kind",
        "parent_lock",
        "forward_patch",
        "changed_parent_locked_files",
        "reconstruction",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("v3-to-reporting source snapshot schema drifted")
    if (
        snapshot["schema_version"] != 1
        or snapshot["snapshot_kind"]
        != "reconstructable_evaluation_v3_to_reporting_v1_source_transition"
    ):
        raise ValueError("wrong v3-to-reporting source snapshot identity")
    parent = snapshot["parent_lock"]
    if not isinstance(parent, dict) or set(parent) != {
        "path",
        "sha256",
        "locked_file_count",
    }:
        raise ValueError("v3-to-reporting snapshot parent schema drifted")
    if (
        _root_relative_file(root, parent["path"], "snapshot v3 parent lock")
        != v3_source
        or parent["sha256"] != file_sha256(v3_source)
        or parent["locked_file_count"] != len(v3_payload["files"])  # type: ignore[arg-type]
    ):
        raise ValueError("v3-to-reporting snapshot parent identity drifted")
    patch = snapshot["forward_patch"]
    if not isinstance(patch, dict) or set(patch) != {
        "path",
        "sha256",
        "format",
        "line_count",
        "byte_count",
    }:
        raise ValueError("v3-to-reporting patch record schema drifted")
    if (
        _root_relative_file(root, patch["path"], "snapshot reporting patch")
        != patch_source
        or patch["sha256"] != file_sha256(patch_source)
        or patch["format"] != "unified_diff_a_to_b_strip_1"
        or patch["line_count"] != len(patch_source.read_bytes().splitlines())
        or patch["byte_count"] != patch_source.stat().st_size
    ):
        raise ValueError("v3-to-reporting patch identity drifted")
    parent_files = v3_payload["files"]
    final_files = reporting_payload["files"]
    if not isinstance(parent_files, dict) or not isinstance(final_files, dict):
        raise ValueError("v3-to-reporting locked file maps are invalid")
    if not set(parent_files).issubset(final_files):
        raise ValueError("reporting lock does not retain every v3 locked path")
    expected_changed = {
        relative
        for relative, digest in parent_files.items()
        if final_files[relative] != digest
    }
    changed = snapshot["changed_parent_locked_files"]
    if not isinstance(changed, dict) or set(changed) != expected_changed:
        raise ValueError("v3-to-reporting changed-file manifest drifted")
    for relative, record in changed.items():
        if not isinstance(record, dict) or set(record) != {
            "v3_sha256",
            "reporting_v1_sha256",
        }:
            raise ValueError("v3-to-reporting changed-file record schema drifted")
        if (
            record["v3_sha256"] != parent_files[relative]
            or record["reporting_v1_sha256"] != final_files[relative]
        ):
            raise ValueError(
                f"v3-to-reporting changed-file hashes drifted: {relative}"
            )
    reconstruction = snapshot["reconstruction"]
    if not isinstance(reconstruction, dict) or set(reconstruction) != {
        "v3_from_reporting_command",
        "reporting_from_v3_command",
        "verify_v3_command",
        "forward_patch_verified",
        "reverse_patch_verified",
    }:
        raise ValueError("v3-to-reporting reconstruction schema drifted")
    if (
        reconstruction["forward_patch_verified"] is not True
        or reconstruction["reverse_patch_verified"] is not True
        or not all(
            str(reconstruction[key]).strip()
            for key in (
                "v3_from_reporting_command",
                "reporting_from_v3_command",
                "verify_v3_command",
            )
        )
    ):
        raise ValueError("v3-to-reporting reconstruction was not verified")


def _load_and_verify_reporting_lock(
    experiment_root: str | Path, lock_path: str | Path
) -> tuple[
    Path,
    Path,
    dict[str, object],
    Path,
    dict[str, object],
    Path,
    dict[str, object],
]:
    """Verify reporting-only code plus immutable v3 producer lineage."""

    root, source, payload = _read_lock_payload(experiment_root, lock_path)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("reporting lock must stay within experiment root") from error
    required = {
        "schema_version",
        "lock_kind",
        "lock_generation",
        "locked_at",
        "locked_after_probe_v3_completion",
        "locked_before_formal_summarizer_retry",
        "parent_lock",
        "finalization_lock",
        "visibility",
        "failed_summarizer_attempts",
        "prelock_state",
        "inherited_inputs",
        "historical_producers",
        "development_provenance",
        "files",
        "execution",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("reporting lock top-level schema drifted")
    if (
        payload["schema_version"] != _REPORTING_LOCK_SCHEMA
        or payload["lock_kind"] != "formal_reporting_retry"
        or payload["lock_generation"]
        != "calibration_identity_recompute_tolerance_v1"
    ):
        raise ValueError("wrong reporting lock identity")
    if (
        not str(payload["locked_at"]).strip()
        or payload["locked_after_probe_v3_completion"] is not True
        or payload["locked_before_formal_summarizer_retry"] is not True
    ):
        raise ValueError("reporting lock timing declarations are inconsistent")

    v3_source = _verify_root_record(root, payload["parent_lock"], "reporting parent lock")
    (
        _,
        verified_v3_source,
        v3_payload,
        v2_source,
        v2_payload,
        _,
        _,
    ) = _load_and_verify_v3_chain(root, v3_source, verify_final_files=False)
    if verified_v3_source != v3_source:
        raise AssertionError("reporting parent v3 source drifted")
    finalization_source = _verify_root_record(
        root, payload["finalization_lock"], "reporting finalization lock"
    )
    if (
        file_sha256(finalization_source) != _FINALIZATION_LOCK_SHA256
        or payload["finalization_lock"] != v3_payload["finalization_lock"]
    ):
        raise ValueError("reporting retry changed FINALIZATION_LOCK.v1")

    visibility = payload["visibility"]
    if not isinstance(visibility, dict) or set(visibility) != {
        "calibration_intercept_values_seen",
        "observed_identity",
        "auroc_auprc_brier_ece_prediction_or_selection_performance_values_seen",
        "revision_used_model_direction_or_performance",
    }:
        raise ValueError("reporting visibility schema drifted")
    expected_observation = {
        "source": "baseline_v2_public_vs_serialized_private_recompute",
        "target": "pCR",
        "model": "dino_vitb16_imagenet1k_mri_clinical_ftv",
        "spatial": "GLOBAL",
        "timing": "T0-T2",
        "analysis_population": "radiomics_complete_case_375",
        "aggregation": "outer_fold_macro",
        "affected_cell_count": 1,
        "column": "calibration_intercept",
        "public_value": "-0.2992681677870132",
        "recomputed_value": "-0.29926816783893295",
        "absolute_difference": "5.191974628004914e-11",
    }
    if (
        visibility["calibration_intercept_values_seen"] is not True
        or visibility["observed_identity"] != expected_observation
        or visibility[
            "auroc_auprc_brier_ece_prediction_or_selection_performance_values_seen"
        ]
        is not False
        or visibility["revision_used_model_direction_or_performance"] is not False
    ):
        raise ValueError("reporting visibility declaration drifted")

    attempts = payload["failed_summarizer_attempts"]
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("reporting lock must record exactly one failed summarizer")
    attempt = attempts[0]
    expected_attempt = {
        "consumer": "summarizer",
        "producer_lock_sha256": file_sha256(v3_source),
        "argument_count": 0,
        "argument_vector_sha256": _EMPTY_ARGUMENT_VECTOR_SHA256,
        "exit_code": 1,
        "expected_public_output_count": 5,
        "public_outputs_written_count": 0,
        "reporting_marker_written": False,
        "exception_type": "ValueError",
        "exception_message": (
            "baseline public aggregate drifted in numeric column "
            "calibration_intercept"
        ),
        "observed_values_limited_to_visibility_record": True,
    }
    if attempt != expected_attempt:
        raise ValueError("failed summarizer provenance drifted")
    state = payload["prelock_state"]
    expected_state = {
        "baseline_v2_completed": True,
        "probe_v3_completed": True,
        "historical_receipts_bound": True,
        "retry_public_outputs_exist": False,
        "reporting_marker_exists": False,
    }
    if state != expected_state:
        raise ValueError("reporting prelock state drifted")
    inherited = payload["inherited_inputs"]
    if inherited != {
        "source": "parent_lock.inherited_inputs",
        "sha256": v3_payload["inherited_inputs"]["sha256"],  # type: ignore[index]
    }:
        raise ValueError("reporting inherited inputs drifted")

    historical = payload["historical_producers"]
    if not isinstance(historical, dict) or set(historical) != {
        "baseline_v2",
        "probe_v3",
    }:
        raise ValueError("reporting historical producer schema drifted")
    if historical["baseline_v2"] != v3_payload["historical_producers"]["baseline_v2"]:  # type: ignore[index]
        raise ValueError("reporting baseline-v2 lineage differs from parent v3")
    probe = historical["probe_v3"]
    if not isinstance(probe, dict) or set(probe) != {
        "consumer",
        "producer_lock",
        "run_receipt",
        "argv",
        "artifacts",
        "audit_counts",
    }:
        raise ValueError("historical probe-v3 schema drifted")
    if probe["consumer"] != "probe":
        raise ValueError("historical producer is not probe-v3")
    probe_lock = _verify_root_record(
        root, probe["producer_lock"], "historical probe-v3 producer lock"
    )
    if probe_lock != v3_source:
        raise ValueError("historical probe producer lock differs from parent v3")
    probe_receipt_source = _verify_root_record(
        root, probe["run_receipt"], "historical probe-v3 run receipt"
    )
    probe_receipt = json.loads(probe_receipt_source.read_text(encoding="utf-8"))
    receipt_required = {
        "schema_version",
        "receipt_kind",
        "consumer",
        "lock_sha256",
        "parent_lock_sha256",
        "argument_count",
        "argument_vector_sha256",
        "artifact_hash_method",
        "metric_values_viewed",
        "artifacts",
    }
    if not isinstance(probe_receipt, dict) or set(probe_receipt) != receipt_required:
        raise ValueError("historical probe-v3 receipt schema drifted")
    argv = probe["argv"]
    if (
        not isinstance(argv, dict)
        or set(argv) != {"argc", "sha256"}
        or probe_receipt["schema_version"] != 1
        or probe_receipt["receipt_kind"] != "formal_metric_free_run_provenance"
        or probe_receipt["consumer"] != "probe"
        or probe_receipt["lock_sha256"] != file_sha256(v3_source)
        or probe_receipt["parent_lock_sha256"] != file_sha256(v2_source)
        or probe_receipt["artifact_hash_method"]
        != "sha256_binary_stream_no_parse"
        or probe_receipt["metric_values_viewed"] is not False
        or probe_receipt["argument_count"] != argv["argc"]
        or probe_receipt["argument_vector_sha256"] != argv["sha256"]
        or argv != v3_payload["execution"]["consumer_argv"]["probe"]  # type: ignore[index]
    ):
        raise ValueError("historical probe-v3 receipt/argv lineage drifted")
    role_to_receipt = {
        "phenotype_predictions": "phenotype_predictions_private",
        "phenotype_selection": "phenotype_selection_private",
        "phenotype_metrics": "phenotype_metrics_public",
        "subtype_predictions": "subtype_predictions_private",
        "subtype_selection": "subtype_selection_private",
        "subtype_metrics": "subtype_metrics_public",
        "ftv_predictions": "ftv_predictions_private",
        "ftv_selection": "ftv_selection_private",
        "ftv_metrics": "ftv_metrics_public",
        "progress": "probe_progress_private",
    }
    artifacts = probe["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(role_to_receipt)
        or not isinstance(probe_receipt["artifacts"], dict)
        or set(probe_receipt["artifacts"]) != set(role_to_receipt.values())
    ):
        raise ValueError("historical probe-v3 artifact schema drifted")
    for role, receipt_key in role_to_receipt.items():
        artifact_path = _verify_root_record(
            root, artifacts[role], f"historical probe-v3 artifact {role}"
        )
        receipt_record = probe_receipt["artifacts"][receipt_key]
        if (
            not isinstance(receipt_record, dict)
            or set(receipt_record) != {"path", "sha256"}
            or receipt_record != artifacts[role]
            or artifact_path
            != _root_relative_file(
                root, receipt_record["path"], f"probe receipt artifact {role}"
            )
        ):
            raise ValueError(f"historical probe-v3 artifact lineage drifted: {role}")
    expected_probe_audit_counts = {
        "phenotype_prediction_rows": 9696,
        "subtype_prediction_rows": 4848,
        "ftv_prediction_rows": 15750,
        "phenotype_selection_rows": 60,
        "subtype_selection_rows": 30,
        "ftv_selection_rows": 210,
        "phenotype_public_rows": 24,
        "subtype_public_rows": 12,
        "ftv_public_rows": 84,
        "progress_fold_completed": 300,
        "progress_candidate_completed": 3300,
        "progress_ovr_estimator_completed": 2160,
        "max_binary_n_iter": 677,
        "max_subtype_n_iter": 197,
        "max_ridge_n_iter": 805,
        "capped_candidate_count": 0,
    }
    if probe["audit_counts"] != expected_probe_audit_counts:
        raise ValueError("historical probe-v3 audited counts drifted")

    development = payload["development_provenance"]
    if not isinstance(development, dict) or set(development) != {
        "source_snapshot",
        "patch_manifest",
        "synthetic_receipt",
    }:
        raise ValueError("reporting development provenance schema drifted")
    snapshot_source = _verify_root_record(
        root, development["source_snapshot"], "v3-to-reporting source snapshot"
    )
    patch_source = _verify_root_record(
        root, development["patch_manifest"], "v3-to-reporting patch manifest"
    )
    _verify_root_record(
        root, development["synthetic_receipt"], "reporting synthetic receipt"
    )
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("reporting lock has no final file hashes")
    for relative, digest in sorted(files.items()):
        path = _root_relative_file(root, relative, f"reporting locked file {relative}")
        expected = _checked_lower_digest(
            digest, f"reporting locked file {relative} SHA-256"
        )
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"reporting file drifted: {relative}; expected={expected}, observed={observed}"
            )
    _verify_v3_to_reporting_source_snapshot(
        root=root,
        snapshot_source=snapshot_source,
        patch_source=patch_source,
        v3_source=v3_source,
        v3_payload=v3_payload,
        reporting_payload=payload,
    )
    execution = payload["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "parent_execution_sha256",
        "consumer_argv",
        "producer_order",
        "marker_summarizer_protocol_version",
    }:
        raise ValueError("reporting execution schema drifted")
    if (
        execution["parent_execution_sha256"]
        != canonical_json_sha256(v3_payload["execution"])
        or execution["consumer_argv"]
        != {
            "summarizer": {
                "argc": 0,
                "sha256": _EMPTY_ARGUMENT_VECTOR_SHA256,
            }
        }
        or execution["producer_order"]
        != ["baseline:v2", "probe:v3", "summarizer:reporting-v1"]
        or execution["marker_summarizer_protocol_version"] != "v3"
    ):
        raise ValueError("reporting execution contract drifted")
    return root, source, payload, v3_source, v3_payload, v2_source, v2_payload


def _verify_reporting_consumer(
    payload: Mapping[str, object],
    expected_consumer: str,
    command_argv: Sequence[str | Path] | None,
) -> tuple[int, str]:
    if expected_consumer != "summarizer":
        raise ValueError("reporting lock consumer must be summarizer")
    if command_argv is None:
        raise ValueError("reporting lock verification requires exact command argv")
    values = tuple(str(value) for value in command_argv)
    digest = argument_vector_sha256(values)
    contract = payload["execution"]["consumer_argv"]["summarizer"]  # type: ignore[index]
    if contract["argc"] != len(values) or contract["sha256"] != digest:
        raise ValueError("summarizer command argv does not match reporting lock")
    return len(values), digest


def _reporting_code_receipt(
    root: Path,
    source: Path,
    payload: Mapping[str, object],
    v3_source: Path,
    *,
    argument_count: int,
    argument_sha256: str,
) -> dict[str, object]:
    development = payload["development_provenance"]
    return {
        "schema_version": _REPORTING_LOCK_SCHEMA,
        "lock_generation": "calibration_identity_recompute_tolerance_v1",
        "lock_path": source.relative_to(root).as_posix(),
        "lock_sha256": file_sha256(source),
        "parent_lock_sha256": file_sha256(v3_source),
        "inherited_inputs_sha256": payload["inherited_inputs"]["sha256"],  # type: ignore[index]
        "verified_file_count": len(payload["files"]),
        "expected_consumer": "summarizer",
        "argument_count": argument_count,
        "argument_vector_sha256": argument_sha256,
        "source_snapshot_sha256": development["source_snapshot"]["sha256"],  # type: ignore[index]
        "patch_manifest_sha256": development["patch_manifest"]["sha256"],  # type: ignore[index]
        "synthetic_receipt_sha256": development["synthetic_receipt"]["sha256"],  # type: ignore[index]
        "finalization_lock_sha256": payload["finalization_lock"]["sha256"],  # type: ignore[index]
        "marker_summarizer_protocol_version": "v3",
        "execution": payload["execution"],
    }


def _verify_locked_formal_inputs(
    *,
    root: Path,
    inputs: object,
    foundation_features: Sequence[str | Path],
    current_cnn_features: Mapping[tuple[str, str], Mapping[int, str | Path]],
    fold_manifest: str | Path,
    clinical_labels: str | Path,
    radiomics: str | Path,
) -> tuple[tuple[str, ...], int]:
    inputs = _validate_v1_input_schema(inputs)
    locked_fold = _verify_record(root, inputs["fold_manifest"], "fold manifest")
    locked_clinical = _verify_record(root, inputs["clinical_labels"], "clinical labels")
    locked_radiomics = _verify_record(root, inputs["radiomics"], "radiomics")
    requested_tabular = (
        Path(fold_manifest).resolve(strict=True),
        Path(clinical_labels).resolve(strict=True),
        Path(radiomics).resolve(strict=True),
    )
    if requested_tabular != (locked_fold, locked_clinical, locked_radiomics):
        raise ValueError(
            "formal tabular path arguments do not match the evaluation lock"
        )

    foundation_records = inputs["foundation_features"]
    if not isinstance(foundation_records, list) or not foundation_records:
        raise ValueError("evaluation lock has no foundation features")
    locked_foundation: list[Path] = []
    foundation_models: list[str] = []
    for index, record in enumerate(foundation_records):
        if not isinstance(record, dict) or set(record) != {"model", "path", "sha256"}:
            raise ValueError("foundation feature lock record schema drifted")
        model = str(record["model"]).strip()
        if not model or model in foundation_models:
            raise ValueError("foundation feature model identity is empty/duplicated")
        foundation_models.append(model)
        locked_foundation.append(
            _verify_record(
                root,
                {"path": record["path"], "sha256": record["sha256"]},
                f"foundation feature {index}/{model}",
            )
        )
    if _normalise_paths(foundation_features) != tuple(
        sorted(locked_foundation, key=str)
    ):
        raise ValueError("formal foundation feature arguments do not match the lock")

    cnn_records = inputs["current_cnn_features"]
    if not isinstance(cnn_records, list) or not cnn_records:
        raise ValueError("evaluation lock has no current-CNN features")
    locked_cnn: dict[tuple[str, str], dict[int, Path]] = {}
    for index, record in enumerate(cnn_records):
        if not isinstance(record, dict) or set(record) != {
            "model",
            "spatial",
            "fold",
            "feature",
            "metadata",
        }:
            raise ValueError("current-CNN feature lock record schema drifted")
        model = str(record["model"]).strip()
        spatial = str(record["spatial"]).strip().upper()
        fold = int(record["fold"])
        if (model, spatial) not in {("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL")}:
            raise ValueError("evaluation lock contains an unapproved current-CNN arm")
        if fold not in range(5) or fold in locked_cnn.setdefault((model, spatial), {}):
            raise ValueError("current-CNN lock fold is invalid/duplicated")
        feature = _verify_record(
            root, record["feature"], f"current-CNN feature {index}"
        )
        metadata = _verify_record(
            root, record["metadata"], f"current-CNN metadata {index}"
        )
        if metadata.parent != feature.parent:
            raise ValueError("current-CNN feature and metadata are not adjacent")
        locked_cnn[(model, spatial)][fold] = feature
    if set(locked_cnn) != {("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL")}:
        raise ValueError("evaluation lock must contain exactly GAP0/LOCAL0")
    if any(set(paths) != set(range(5)) for paths in locked_cnn.values()):
        raise ValueError("evaluation lock must contain all five current-CNN folds")
    requested_cnn = {
        (str(model), str(spatial).upper()): {
            int(fold): Path(path).resolve(strict=True) for fold, path in paths.items()
        }
        for (model, spatial), paths in current_cnn_features.items()
    }
    if requested_cnn != locked_cnn:
        raise ValueError("formal current-CNN feature arguments do not match the lock")

    return tuple(foundation_models), sum(len(value) for value in locked_cnn.values())


def verify_evaluation_code_lock(
    *,
    experiment_root: str | Path,
    lock_path: str | Path,
    expected_consumer: str = "summarizer",
    command_argv: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """Verify v1 code, or the complete v2 parent/final-code chain.

    Schema-v2 callers must pass the exact argument-parser vector (excluding the
    interpreter and script path).  This API is the reporting/summarizer gate.
    """

    _, _, peek = _read_lock_payload(experiment_root, lock_path)
    if peek.get("schema_version") == 1:
        _, source, payload = _load_and_verify_code_lock(experiment_root, lock_path)
        return {
            "lock_path": str(source),
            "lock_sha256": file_sha256(source),
            "verified_file_count": len(payload["files"]),
            "execution": payload["execution"],
        }
    if peek.get("schema_version") == 2:
        root, source, payload, parent_source, _ = _load_and_verify_v2_chain(
            experiment_root, lock_path
        )
        count, digest = _verify_v2_consumer(payload, expected_consumer, command_argv)
        return _v2_code_receipt(
            root,
            source,
            payload,
            parent_source,
            expected_consumer=expected_consumer,
            argument_count=count,
            argument_sha256=digest,
        )
    if peek.get("schema_version") == 3:
        root, source, payload, v2_source, _, _, _ = _load_and_verify_v3_chain(
            experiment_root, lock_path
        )
        count, digest = _verify_v3_consumer(payload, expected_consumer, command_argv)
        return _v3_code_receipt(
            root,
            source,
            payload,
            v2_source,
            expected_consumer=expected_consumer,
            argument_count=count,
            argument_sha256=digest,
        )
    if peek.get("schema_version") == _REPORTING_LOCK_SCHEMA:
        root, source, payload, v3_source, _, _, _ = (
            _load_and_verify_reporting_lock(experiment_root, lock_path)
        )
        count, digest = _verify_reporting_consumer(
            payload, expected_consumer, command_argv
        )
        return _reporting_code_receipt(
            root,
            source,
            payload,
            v3_source,
            argument_count=count,
            argument_sha256=digest,
        )
    raise ValueError("unsupported evaluation lock schema")


def verify_formal_evaluation_lock(
    *,
    experiment_root: str | Path,
    lock_path: str | Path,
    foundation_features: Sequence[str | Path],
    current_cnn_features: Mapping[tuple[str, str], Mapping[int, str | Path]],
    fold_manifest: str | Path,
    clinical_labels: str | Path,
    radiomics: str | Path,
    expected_consumer: str | None = None,
    command_argv: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """Verify final code and the exact locked formal input/CLI argument set.

    Existing schema-v1 calls remain valid.  Schema v2 inherits the byte-locked
    parent v1 input records while binding the caller to final v2 code and argv.
    """

    _, _, peek = _read_lock_payload(experiment_root, lock_path)
    if peek.get("schema_version") == 1:
        root, source, payload = _load_and_verify_code_lock(experiment_root, lock_path)
        models, cnn_assets = _verify_locked_formal_inputs(
            root=root,
            inputs=payload["inputs"],
            foundation_features=foundation_features,
            current_cnn_features=current_cnn_features,
            fold_manifest=fold_manifest,
            clinical_labels=clinical_labels,
            radiomics=radiomics,
        )
        execution = payload["execution"]
        if not isinstance(execution, dict):
            raise ValueError("evaluation execution lock must be an object")
        return {
            "lock_path": str(source),
            "lock_sha256": file_sha256(source),
            "verified_file_count": len(payload["files"]),
            "foundation_models": models,
            "current_cnn_assets": cnn_assets,
            "execution": execution,
        }
    if peek.get("schema_version") == 2:
        if expected_consumer not in _V2_RUN_CONSUMERS:
            raise ValueError(
                "schema-v2 formal evaluation consumer must be baseline or probe"
            )
        root, source, payload, parent_source, parent_payload = _load_and_verify_v2_chain(
            experiment_root, lock_path
        )
        count, digest = _verify_v2_consumer(payload, expected_consumer, command_argv)
        models, cnn_assets = _verify_locked_formal_inputs(
            root=root,
            inputs=parent_payload["inputs"],
            foundation_features=foundation_features,
            current_cnn_features=current_cnn_features,
            fold_manifest=fold_manifest,
            clinical_labels=clinical_labels,
            radiomics=radiomics,
        )
        receipt = _v2_code_receipt(
            root,
            source,
            payload,
            parent_source,
            expected_consumer=expected_consumer,
            argument_count=count,
            argument_sha256=digest,
        )
    elif peek.get("schema_version") == 3:
        if expected_consumer not in _V3_RUN_CONSUMERS:
            raise ValueError("schema-v3 formal evaluation consumer must be probe")
        root, source, payload, v2_source, _, _, parent_payload = (
            _load_and_verify_v3_chain(experiment_root, lock_path)
        )
        count, digest = _verify_v3_consumer(payload, expected_consumer, command_argv)
        models, cnn_assets = _verify_locked_formal_inputs(
            root=root,
            inputs=parent_payload["inputs"],
            foundation_features=foundation_features,
            current_cnn_features=current_cnn_features,
            fold_manifest=fold_manifest,
            clinical_labels=clinical_labels,
            radiomics=radiomics,
        )
        receipt = _v3_code_receipt(
            root,
            source,
            payload,
            v2_source,
            expected_consumer=expected_consumer,
            argument_count=count,
            argument_sha256=digest,
        )
        if tuple(models) != _FORMAL_FOUNDATION_MODELS:
            raise ValueError(
                "formal foundation feature set/order must be exactly MedicalNet then DINO"
            )
    else:
        raise ValueError("unsupported evaluation lock schema")
    receipt.update(
        {
            "foundation_models": models,
            "current_cnn_assets": cnn_assets,
        }
    )
    return receipt


def _artifact_records(
    root: Path, artifacts: Mapping[str, str | Path]
) -> dict[str, dict[str, str]]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("run provenance requires at least one artifact")
    records: dict[str, dict[str, str]] = {}
    resolved_paths: set[Path] = set()
    for raw_key, raw_path in sorted(artifacts.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if not _ARTIFACT_KEY_RE.fullmatch(key):
            raise ValueError("artifact keys must be lowercase snake_case identifiers")
        if key in records:
            raise ValueError("run provenance artifact keys are duplicated")
        path = _path_within_root(root, raw_path, f"artifact {key}", must_exist=True)
        if path in resolved_paths:
            raise ValueError("run provenance artifact paths are duplicated")
        resolved_paths.add(path)
        records[key] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
    return records


def _atomic_write_private_json(
    destination: Path, payload: Mapping[str, object]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"run provenance receipt already exists: {destination.name}"
        )
    rendered = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and refuses an existing destination.
        os.link(temporary, destination)
        os.chmod(destination, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_metric_free_run_provenance(
    *,
    experiment_root: str | Path,
    lock_path: str | Path,
    expected_consumer: str,
    command_argv: Sequence[str | Path],
    artifacts: Mapping[str, str | Path],
    receipt_path: str | Path,
) -> dict[str, object]:
    """Hash formal artifacts without parsing them and atomically publish a receipt.

    Only the locked baseline and probe producers may write these receipts.  CSV
    values are never parsed or selected; ``file_sha256`` streams raw bytes.
    """

    root = Path(experiment_root).resolve(strict=True)
    _, _, peek = _read_lock_payload(root, lock_path)
    allowed = _V3_RUN_CONSUMERS if peek.get("schema_version") == 3 else _V2_RUN_CONSUMERS
    if expected_consumer not in allowed:
        raise ValueError("metric-free provenance consumer is invalid for this lock")
    destination = _path_within_root(
        root, receipt_path, "run provenance receipt", must_exist=False
    )
    code_receipt = verify_evaluation_code_lock(
        experiment_root=root,
        lock_path=lock_path,
        expected_consumer=expected_consumer,
        command_argv=command_argv,
    )
    if code_receipt.get("schema_version") not in {2, 3}:
        raise ValueError("metric-free run provenance requires a chained lock")
    artifact_records = _artifact_records(root, artifacts)
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_kind": "formal_metric_free_run_provenance",
        "consumer": expected_consumer,
        "lock_sha256": code_receipt["lock_sha256"],
        "parent_lock_sha256": code_receipt["parent_lock_sha256"],
        "argument_count": code_receipt["argument_count"],
        "argument_vector_sha256": code_receipt["argument_vector_sha256"],
        "artifact_hash_method": "sha256_binary_stream_no_parse",
        "metric_values_viewed": False,
        "artifacts": artifact_records,
    }
    _atomic_write_private_json(destination, payload)
    return {
        **payload,
        "receipt_path": destination.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(destination),
    }


def verify_metric_free_run_provenance(
    *,
    experiment_root: str | Path,
    lock_path: str | Path,
    receipt_path: str | Path,
    expected_consumer: str,
    expected_artifacts: Mapping[str, str | Path],
) -> dict[str, object]:
    """Verify producer/code/argv lineage and artifact bytes before CSV parsing."""

    root = Path(experiment_root).resolve(strict=True)
    _, _, peek = _read_lock_payload(root, lock_path)
    allowed = _V3_RUN_CONSUMERS if peek.get("schema_version") == 3 else _V2_RUN_CONSUMERS
    if expected_consumer not in allowed:
        raise ValueError("metric-free provenance consumer is invalid for this lock")
    source = _path_within_root(
        root, receipt_path, "run provenance receipt", must_exist=True
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "receipt_kind",
        "consumer",
        "lock_sha256",
        "parent_lock_sha256",
        "argument_count",
        "argument_vector_sha256",
        "artifact_hash_method",
        "metric_values_viewed",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("metric-free run provenance schema drifted")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported metric-free run provenance schema")
    if payload["receipt_kind"] != "formal_metric_free_run_provenance":
        raise ValueError("wrong metric-free run provenance kind")
    if payload["consumer"] != expected_consumer:
        raise ValueError("run provenance producer does not match expected consumer")
    if payload["artifact_hash_method"] != "sha256_binary_stream_no_parse":
        raise ValueError("run provenance did not use the metric-free hash method")
    if payload["metric_values_viewed"] is not False:
        raise ValueError("run provenance claims metric values were viewed")
    lock_digest = _checked_lower_digest(payload["lock_sha256"], "receipt lock SHA-256")
    parent_digest = _checked_lower_digest(
        payload["parent_lock_sha256"], "receipt parent lock SHA-256"
    )
    argument_digest = _checked_lower_digest(
        payload["argument_vector_sha256"], "receipt argv SHA-256"
    )
    argument_count = payload["argument_count"]
    if isinstance(argument_count, bool) or not isinstance(argument_count, int):
        raise ValueError("receipt argv count must be a nonnegative integer")
    if argument_count < 0:
        raise ValueError("receipt argv count must be a nonnegative integer")

    if peek.get("schema_version") == 2:
        _, lock_source, lock_payload, parent_source, _ = _load_and_verify_v2_chain(
            root, lock_path
        )
    elif peek.get("schema_version") == 3:
        _, lock_source, lock_payload, parent_source, _, _, _ = _load_and_verify_v3_chain(
            root, lock_path
        )
    else:
        raise ValueError("metric-free provenance requires a chained lock")
    consumer_contract = lock_payload["execution"]["consumer_argv"][  # type: ignore[index]
        expected_consumer
    ]
    if (
        lock_digest != file_sha256(lock_source)
        or parent_digest != file_sha256(parent_source)
        or argument_count != consumer_contract["argc"]
        or argument_digest != consumer_contract["sha256"]
    ):
        raise ValueError("run provenance lock/parent/argv chain does not match")

    expected_records = _artifact_records(root, expected_artifacts)
    records = payload["artifacts"]
    if not isinstance(records, dict) or set(records) != set(expected_records):
        raise ValueError("run provenance artifact set does not match")
    verified: dict[str, str] = {}
    for key, expected in expected_records.items():
        record = records[key]
        path = _verify_root_record(root, record, f"receipt artifact {key}")
        if path != _root_relative_file(
            root, expected["path"], f"expected artifact {key}"
        ):
            raise ValueError(f"run provenance artifact path does not match: {key}")
        if record["sha256"] != expected["sha256"]:
            raise ValueError(f"run provenance artifact SHA-256 does not match: {key}")
        verified[key] = str(record["sha256"])
    return {
        "schema_version": 1,
        "consumer": expected_consumer,
        "lock_sha256": lock_digest,
        "parent_lock_sha256": parent_digest,
        "argument_vector_sha256": argument_digest,
        "artifact_count": len(verified),
        "artifact_sha256": verified,
        "receipt_path": source.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(source),
    }


def verify_historical_metric_free_run_provenance(
    *,
    experiment_root: str | Path,
    active_lock_path: str | Path,
    producer_key: str,
    expected_artifacts: Mapping[str, str | Path],
) -> dict[str, object]:
    """Verify a historical producer named only by the active chained lock.

    The caller cannot provide or substitute the historical lock/receipt paths.
    Artifact files are streamed only for SHA-256 and are never parsed.
    """

    _, _, peek = _read_lock_payload(experiment_root, active_lock_path)
    if peek.get("schema_version") == 3:
        if producer_key != "baseline_v2":
            raise ValueError("schema-v3 historical producer_key must be baseline_v2")
        root, _, payload, _, _, _, _ = _load_and_verify_v3_chain(
            experiment_root, active_lock_path
        )
        schema_version: object = 3
        expected_keys = {"baseline_v2"}
    elif peek.get("schema_version") == _REPORTING_LOCK_SCHEMA:
        root, _, payload, _, _, _, _ = _load_and_verify_reporting_lock(
            experiment_root, active_lock_path
        )
        schema_version = _REPORTING_LOCK_SCHEMA
        expected_keys = {"baseline_v2", "probe_v3"}
    else:
        raise ValueError("historical provenance requires v3/reporting chained lock")
    if producer_key not in expected_keys:
        raise ValueError("historical producer_key is not allowed by the active lock")
    producer = payload["historical_producers"][producer_key]  # type: ignore[index]
    artifacts = producer["artifacts"]
    if not isinstance(expected_artifacts, Mapping) or set(expected_artifacts) != set(
        artifacts
    ):
        raise ValueError("historical expected artifact set does not match active lock")
    verified: dict[str, str] = {}
    for role, raw_path in expected_artifacts.items():
        expected_path = _path_within_root(
            root, raw_path, f"expected historical artifact {role}", must_exist=True
        )
        locked_record = artifacts[role]
        locked_path = _verify_root_record(
            root, locked_record, f"historical {producer_key} artifact {role}"
        )
        if expected_path != locked_path:
            raise ValueError(f"historical artifact path differs: {producer_key}/{role}")
        verified[str(role)] = str(locked_record["sha256"])
    receipt_path = _verify_root_record(
        root, producer["run_receipt"], f"historical {producer_key} run receipt"
    )
    producer_lock = _verify_root_record(
        root, producer["producer_lock"], f"historical {producer_key} producer lock"
    )
    return {
        "schema_version": schema_version,
        "consumer": producer["consumer"],
        "lock_generation": (
            "whole_evaluation_v2"
            if producer_key == "baseline_v2"
            else "probe_retry_v3"
        ),
        "lock_sha256": file_sha256(producer_lock),
        "argument_vector_sha256": producer["argv"]["sha256"],
        "artifact_count": len(verified),
        "artifact_sha256": verified,
        "receipt_sha256": file_sha256(receipt_path),
    }


__all__ = [
    "argument_vector_sha256",
    "canonical_json_sha256",
    "file_sha256",
    "verify_evaluation_code_lock",
    "verify_formal_evaluation_lock",
    "verify_historical_metric_free_run_provenance",
    "verify_metric_free_run_provenance",
    "write_metric_free_run_provenance",
]
