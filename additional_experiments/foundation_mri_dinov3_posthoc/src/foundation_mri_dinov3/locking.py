"""Small fail-closed locks for the isolated post-hoc extension.

The original experiment has its own immutable lock chain.  This module never
tries to reinterpret it: it verifies the published bytes as parent evidence,
then verifies extension-local manifests and exact empty/default command lines.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .paths import EXPERIMENT_ROOT, REPOSITORY_ROOT


MODEL_INPUT_LOCK = EXPERIMENT_ROOT / "configs/MODEL_INPUT_LOCK.v1.json"
EVALUATION_LOCK = EXPERIMENT_ROOT / "configs/EVALUATION_LOCK.v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONSUMERS = ("baseline", "probe", "report")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def argv_sha256(argv: Sequence[str]) -> str:
    return canonical_json_sha256([str(value) for value in argv])


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"lock input must be a regular file: {source}")
    value = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _digest(value: object, label: str) -> str:
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _resolve_manifest_path(raw: object, *, repository_root: Path) -> Path:
    text = str(raw)
    path = Path(text)
    if path.is_absolute():
        return Path(os.path.abspath(path))
    resolved = Path(os.path.abspath(repository_root / path))
    try:
        resolved.relative_to(Path(os.path.abspath(repository_root)))
    except ValueError as exc:
        raise ValueError(f"relative manifest path escapes repository: {text}") from exc
    return resolved


def verify_hash_records(
    records: Mapping[str, object],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    """Verify ``role -> {path,sha256}`` records without parsing their values."""

    if not isinstance(records, Mapping) or not records:
        raise ValueError("hash record mapping must be non-empty")
    observed: dict[str, str] = {}
    for role, raw in sorted(records.items()):
        if not isinstance(raw, Mapping) or set(raw) not in (
            {"path", "sha256"},
            {"path", "sha256", "bytes"},
        ):
            raise ValueError(f"invalid hash record schema for {role}")
        path = _resolve_manifest_path(raw["path"], repository_root=repository_root)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing regular manifest file for {role}: {path}")
        if "bytes" in raw:
            expected_bytes = raw["bytes"]
            if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
                raise ValueError(f"bytes must be an integer for {role}")
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"byte-size drift for {role}")
        expected = _digest(raw["sha256"], f"{role}.sha256")
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(
                f"SHA-256 drift for {role}: expected={expected}, observed={digest}"
            )
        observed[str(role)] = digest
    return observed


def _require_read_only_lock(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if path.stat().st_mode & 0o222:
        raise PermissionError(f"formal lock must be read-only: {path}")


def _verify_parent_publication(
    lock: Mapping[str, Any], *, repository_root: Path
) -> None:
    parent = lock.get("parent_publication")
    if not isinstance(parent, Mapping) or set(parent) != {"path", "sha256"}:
        raise ValueError("parent_publication must contain path and sha256")
    verify_hash_records(
        {"parent_publication": parent}, repository_root=repository_root
    )
    parent_document = load_json(
        _resolve_manifest_path(parent["path"], repository_root=repository_root)
    )
    if parent_document.get("original_candidate_set_is_immutable") is not True:
        raise ValueError("parent publication does not preserve the original candidate set")
    if parent_document.get("original_results_were_visible_before_extension") is not True:
        raise ValueError("post-hoc visibility disclosure is missing")
    verify_hash_records(
        parent_document.get("artifacts", {}), repository_root=repository_root
    )


def verify_model_input_lock(
    snapshot_dir: str | Path,
    command_argv: Sequence[str] | None = None,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    lock_path: Path | None = None,
    require_environment: bool = True,
) -> Mapping[str, Any]:
    """Verify the pre-extraction lock and the three official HF artifacts."""

    source = lock_path or (experiment_root / "configs/MODEL_INPUT_LOCK.v1.json")
    _require_read_only_lock(source)
    lock = load_json(source)
    required = {
        "schema_version",
        "status",
        "created_utc",
        "post_hoc_disclosure",
        "formal_dinov3_outcomes_loaded",
        "parent_publication",
        "protocol",
        "locked_files",
        "model_artifacts",
        "upstream_inputs",
        "required_environment",
        "formal_command_argv",
        "expected_output",
    }
    if set(lock) != required:
        raise ValueError(f"MODEL_INPUT_LOCK schema drifted: {sorted(set(lock) ^ required)}")
    if lock["schema_version"] != "foundation_mri_dinov3_model_input_lock_v1":
        raise ValueError("unexpected model-input lock schema")
    if lock["status"] != "FROZEN_BEFORE_DINOV3_PATIENT_EXTRACTION":
        raise ValueError("model-input lock has the wrong status")
    if lock["formal_dinov3_outcomes_loaded"] is not False:
        raise ValueError("model-input lock was not outcome blind")
    disclosure = lock["post_hoc_disclosure"]
    if not isinstance(disclosure, Mapping) or disclosure.get("no_preregistration_claim") is not True:
        raise ValueError("post-hoc disclosure is incomplete")
    if disclosure.get("patient_level_pretraining_contamination_status") != "unknown":
        raise ValueError("DINOv3 contamination uncertainty must be explicit")
    _verify_parent_publication(lock, repository_root=repository_root)
    verify_hash_records(
        {"protocol": lock["protocol"]}, repository_root=repository_root
    )
    verify_hash_records(lock["locked_files"], repository_root=repository_root)
    verify_hash_records(lock["upstream_inputs"], repository_root=repository_root)

    snapshot = Path(snapshot_dir).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    model_records = lock["model_artifacts"]
    if not isinstance(model_records, Mapping) or set(model_records) != {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }:
        raise ValueError("model artifact set drifted")
    for filename, raw in model_records.items():
        if not isinstance(raw, Mapping) or set(raw) != {"sha256", "bytes"}:
            raise ValueError(f"invalid model artifact record: {filename}")
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes = raw["bytes"]
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"invalid byte count for {filename}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"model artifact byte-size drift: {filename}")
        if file_sha256(path) != _digest(raw["sha256"], f"{filename}.sha256"):
            raise ValueError(f"model artifact SHA-256 drift: {filename}")

    environment = lock["required_environment"]
    if not isinstance(environment, Mapping) or not environment:
        raise ValueError("required_environment must be non-empty")
    if require_environment:
        for key, expected in environment.items():
            if os.environ.get(str(key)) != str(expected):
                raise RuntimeError(f"required offline environment mismatch: {key}")
        configured = os.environ.get("DINOV3_SNAPSHOT_DIR")
        if configured is None or Path(configured).expanduser().resolve() != snapshot:
            raise RuntimeError("DINOV3_SNAPSHOT_DIR does not match the verified snapshot")
    expected_argv = tuple(str(value) for value in lock["formal_command_argv"])
    if command_argv is not None and tuple(str(value) for value in command_argv) != expected_argv:
        raise ValueError("formal extraction argv drifted")
    return {
        "lock_path": source,
        "lock_sha256": file_sha256(source),
        "snapshot_revision": lock["expected_output"]["huggingface_revision"],
        "checkpoint_sha256": model_records["model.safetensors"]["sha256"],
    }


def verify_evaluation_lock(
    consumer: str,
    command_argv: Sequence[str],
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    lock_path: Path | None = None,
) -> Mapping[str, Any]:
    """Verify the outcome-blind feature/evaluation/reporting lock."""

    name = str(consumer)
    if name not in CONSUMERS:
        raise ValueError(f"consumer must be one of {CONSUMERS}")
    source = lock_path or (experiment_root / "configs/EVALUATION_LOCK.v1.json")
    _require_read_only_lock(source)
    lock = load_json(source)
    required = {
        "schema_version",
        "status",
        "created_utc",
        "prior_visibility",
        "parent_model_input_lock",
        "parent_publication",
        "protocols",
        "locked_files",
        "runtime_inputs",
        "feature_asset",
        "parent_comparator_artifacts",
        "commands",
        "expected_counts",
        "exclusive_outputs",
    }
    if set(lock) != required:
        raise ValueError(f"EVALUATION_LOCK schema drifted: {sorted(set(lock) ^ required)}")
    if lock["schema_version"] != "foundation_mri_dinov3_evaluation_lock_v1":
        raise ValueError("unexpected evaluation-lock schema")
    if lock["status"] != "FROZEN_BEFORE_DINOV3_OUTCOME_EVALUATION":
        raise ValueError("evaluation lock has the wrong status")
    visibility = lock["prior_visibility"]
    if not isinstance(visibility, Mapping):
        raise ValueError("prior_visibility must be an object")
    gates = {
        "original_study_outcomes_public": True,
        "extension_is_post_hoc": True,
        "no_preregistration_claim": True,
        "dinov3_outcome_metrics_seen": False,
    }
    if any(visibility.get(key) is not value for key, value in gates.items()):
        raise ValueError("evaluation visibility disclosure drifted")
    _verify_parent_publication(lock, repository_root=repository_root)
    verify_hash_records(
        {"model_input_lock": lock["parent_model_input_lock"]},
        repository_root=repository_root,
    )
    verify_hash_records(lock["protocols"], repository_root=repository_root)
    verify_hash_records(lock["locked_files"], repository_root=repository_root)
    verify_hash_records(lock["runtime_inputs"], repository_root=repository_root)
    verify_hash_records(
        {"feature_asset": lock["feature_asset"]}, repository_root=repository_root
    )
    verify_hash_records(
        lock["parent_comparator_artifacts"], repository_root=repository_root
    )
    commands = lock["commands"]
    if not isinstance(commands, Mapping) or set(commands) != set(CONSUMERS):
        raise ValueError("formal command set drifted")
    expected = tuple(str(value) for value in commands[name])
    if tuple(str(value) for value in command_argv) != expected:
        raise ValueError(f"formal {name} argv drifted")
    counts = lock["expected_counts"]
    if not isinstance(counts, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("expected_counts must contain positive integers")
    return {
        "lock_path": source,
        "lock_sha256": file_sha256(source),
        "consumer": name,
        "argv_sha256": argv_sha256(expected),
        "expected_counts": dict(counts),
        "feature_asset": dict(lock["feature_asset"]),
        "exclusive_outputs": dict(lock["exclusive_outputs"]),
    }


def _relative_public_path(path: Path, *, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("receipt artifact lies outside the repository") from exc


def _atomic_exclusive_json(path: Path, value: Mapping[str, Any], *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    mode = 0o600 if private else 0o644
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(handle, mode)
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_json_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(path) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def write_producer_receipt(
    *,
    consumer: str,
    command_argv: Sequence[str],
    artifacts: Mapping[str, str | Path],
    counts: Mapping[str, int],
    receipt_path: str | Path,
    experiment_root: Path = EXPERIMENT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
) -> Mapping[str, Any]:
    """Write a metric-free private completion marker after all artifacts exist."""

    verified = verify_evaluation_lock(
        consumer,
        command_argv,
        experiment_root=experiment_root,
        repository_root=repository_root,
    )
    if consumer not in {"baseline", "probe"}:
        raise ValueError("only baseline/probe producers write private receipts")
    artifact_records: dict[str, dict[str, object]] = {}
    for role, raw_path in sorted(artifacts.items()):
        path = Path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        artifact_records[str(role)] = {
            "path": _relative_public_path(path, repository_root=repository_root),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    clean_counts: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("producer receipt counts must be positive integers")
        clean_counts[str(key)] = value
    receipt = {
        "schema_version": "foundation_mri_dinov3_producer_receipt_v1",
        "consumer": consumer,
        "evaluation_lock_sha256": verified["lock_sha256"],
        "argv_sha256": verified["argv_sha256"],
        "metric_values_viewed": False,
        "artifact_hash_method": "sha256_binary_stream_no_parse",
        "artifacts": artifact_records,
        "counts": clean_counts,
    }
    _atomic_exclusive_json(Path(receipt_path), receipt, private=True)
    return receipt


def verify_producer_receipt(
    consumer: str,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
) -> Mapping[str, Any]:
    name = str(consumer)
    if name not in {"baseline", "probe"}:
        raise ValueError("producer receipt consumer must be baseline or probe")
    path = experiment_root / f"metrics/{name}_run.private.provenance.json"
    receipt = load_json(path)
    required = {
        "schema_version",
        "consumer",
        "evaluation_lock_sha256",
        "argv_sha256",
        "metric_values_viewed",
        "artifact_hash_method",
        "artifacts",
        "counts",
    }
    if set(receipt) != required:
        raise ValueError("producer receipt schema drifted")
    if receipt["schema_version"] != "foundation_mri_dinov3_producer_receipt_v1":
        raise ValueError("unexpected producer receipt schema")
    if receipt["consumer"] != name or receipt["metric_values_viewed"] is not False:
        raise ValueError("producer receipt identity/visibility drifted")
    lock = verify_evaluation_lock(name, (), experiment_root=experiment_root, repository_root=repository_root)
    if receipt["evaluation_lock_sha256"] != lock["lock_sha256"]:
        raise ValueError("producer receipt evaluation lock drifted")
    if receipt["argv_sha256"] != lock["argv_sha256"]:
        raise ValueError("producer receipt argv drifted")
    verify_hash_records(receipt["artifacts"], repository_root=repository_root)
    return receipt


__all__ = [
    "EVALUATION_LOCK",
    "MODEL_INPUT_LOCK",
    "argv_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "load_json",
    "verify_evaluation_lock",
    "verify_hash_records",
    "verify_model_input_lock",
    "verify_producer_receipt",
    "write_producer_receipt",
]
