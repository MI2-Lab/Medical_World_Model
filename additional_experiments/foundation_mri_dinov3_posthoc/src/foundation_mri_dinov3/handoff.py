"""Outcome-blind, post-commit Git handoff for the DINOv3 extension.

The scientific reporter publishes ``final_report.md`` first and binds those
bytes in its reporting commit marker.  After that scientific bundle has been
committed and a substantive push has been attempted, :func:`create_git_handoff`
records the Git result without reading any prediction or outcome file.
:func:`finalize_handoff` then replaces one fixed HTML marker in the committed
scientific report and publishes a coverage receipt last.

The source-report hash in the reporting marker remains authoritative for the
scientific bytes.  The coverage receipt binds that source hash, the augmented
report hash, and the handoff-manifest hash, avoiding a self-referential commit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any

from .paths import REPOSITORY_ROOT


EXPERIMENT_PREFIX = "additional_experiments/foundation_mri_dinov3_posthoc"
EXPECTED_BRANCH = "feature/foundation-mri-dinov3-posthoc"
EXPECTED_REMOTE = "origin"
EXPECTED_REMOTE_REF = f"refs/heads/{EXPECTED_BRANCH}"

HANDOFF_SCHEMA_VERSION = "foundation_mri_dinov3_git_handoff_v1"
COVERAGE_SCHEMA_VERSION = "foundation_mri_dinov3_final_report_coverage_v1"
REPORTING_SCHEMA_VERSION = "foundation_mri_dinov3_posthoc_reporting_provenance_v1"
EXPECTED_MODEL_NAME = "dinov3_vitb16_lvd1689m_posthoc"
HANDOFF_MARKER = "<!-- FOUNDATION_MRI_DINOV3_GIT_HANDOFF_V1 -->"

PUSH_OK = "SUBSTANTIVE_PUSH_OK"
PUSH_FAILED = "SUBSTANTIVE_PUSH_FAILED"
PUSH_STATUSES = (PUSH_OK, PUSH_FAILED)
MAXIMUM_SANITIZED_ERROR_CHARACTERS = 1000

SCIENTIFIC_PUBLIC_ARTIFACTS: Mapping[str, str] = {
    "paired_comparisons": f"{EXPERIMENT_PREFIX}/metrics/paired_bootstrap_comparisons.csv",
    "results_summary": f"{EXPERIMENT_PREFIX}/metrics/results_summary.json",
    "final_report": f"{EXPERIMENT_PREFIX}/reports/final_report.md",
    "timing_figure": f"{EXPERIMENT_PREFIX}/figures/pcr_timing_performance.png",
    "comparison_figure": f"{EXPERIMENT_PREFIX}/figures/paired_comparison_deltas.png",
}
REPORTING_MARKER_PATH = f"{EXPERIMENT_PREFIX}/metrics/reporting_run_provenance.json"
EVALUATION_LOCK_PATH = f"{EXPERIMENT_PREFIX}/configs/EVALUATION_LOCK.v1.json"
GIT_HANDOFF_PATH = f"{EXPERIMENT_PREFIX}/reports/git_handoff.json"
COVERAGE_PATH = f"{EXPERIMENT_PREFIX}/metrics/final_report_coverage.json"

FROZEN_HANDOFF_SOURCES: Mapping[str, str] = {
    "final_report_template": f"{EXPERIMENT_PREFIX}/reports/final_report.template.md",
    "handoff_module": f"{EXPERIMENT_PREFIX}/src/foundation_mri_dinov3/handoff.py",
    "finalize_handoff_cli": f"{EXPERIMENT_PREFIX}/scripts/finalize_handoff.py",
    "handoff_test": f"{EXPERIMENT_PREFIX}/tests/test_handoff.py",
}

# Every path here must already exist in the substantive content commit.  The
# list includes the complete public reporting bundle and the small finalizer
# implementation that interprets it.  It intentionally contains no private
# prediction, selection, feature, or progress artifact.
TRACKED_CONTENT_PATHS = (
    EVALUATION_LOCK_PATH,
    REPORTING_MARKER_PATH,
    *SCIENTIFIC_PUBLIC_ARTIFACTS.values(),
    *FROZEN_HANDOFF_SOURCES.values(),
)

_HANDOFF_KEYS = {
    "schema_version",
    "content_commit_sha",
    "branch",
    "remote",
    "remote_ref",
    "substantive_push_status",
    "substantive_remote_ref_sha",
    "sanitized_push_error",
    "artifact_sha256",
}
_COVERAGE_KEYS = {
    "schema_version",
    "publication_commit_marker",
    "source_report_sha256",
    "final_report_sha256",
    "git_handoff_sha256",
    "reporting_run_provenance_sha256",
    "content_commit_sha",
    "branch",
    "remote",
    "remote_ref",
    "substantive_push_status",
    "substantive_remote_ref_sha",
    "private_inputs_read",
}
_REPORTING_KEYS = {
    "schema_version",
    "summary_schema_version",
    "model_name",
    "posthoc",
    "comparison_contract_sha256",
    "comparison_spec_count",
    "paired_metric_row_count",
    "input_sha256",
    "public_artifact_sha256",
    "published_last",
    "lineage_mode",
    "report_lock",
    "producers",
}
_EVALUATION_LOCK_KEYS = {
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
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S+")
_TOKEN = re.compile(r"(?i)(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+)")
_AUTH = re.compile(
    r"(?i)\b(?:authorization|bearer|token|password|passwd|secret)\s*[:=]\s*\S+"
)
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:[\\/][^\s\"'<>|]+")
_POSIX_PATH = re.compile(r"(?<![:A-Za-z0-9_.-])/(?:[^\s\"'<>|]+)")
_FORBIDDEN_ERROR = re.compile(
    r"(?i)(?:github_pat_|gh[pousr]_|authorization\s*[:=]|bearer\s+|"
    r"(?:token|password|passwd|secret)\s*[:=]|\b[A-Z]:[\\/]|"
    r"(?<![:A-Za-z0-9_.-])/(?:[^\s\"'<>|]+))"
)

GitRunner = Callable[[Sequence[str], Path], bytes]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _checked_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _checked_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return value


def _resolved_root(repository_root: Path | str) -> Path:
    supplied = Path(repository_root).absolute()
    if supplied.is_symlink():
        raise FileNotFoundError("repository root must not be a symlink")
    root = supplied.resolve()
    if not root.is_dir():
        raise FileNotFoundError("repository root is not a real directory")
    return root


def _fixed_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("formal handoff path is not repository-relative")
    path = root / relative
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved != path:
        raise ValueError("formal handoff path escaped the repository or uses a symlink")
    return path


def _read_public_bytes(root: Path, relative: str, *, label: str) -> bytes:
    if ".private." in relative or any(
        part == "private" for part in Path(relative).parts
    ):
        raise ValueError(f"{label} resolves to a private path")
    path = _fixed_path(root, relative)
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"required public artifact is missing: {relative}"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"required public artifact is not a regular file: {relative}")
    return path.read_bytes()


def _default_git_runner(arguments: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        # Git stderr can contain credential-bearing URLs or local paths.  It is
        # deliberately not interpolated into this verification exception.
        command = arguments[0] if arguments else "command"
        raise ValueError(
            f"git {command} verification failed (exit {completed.returncode})"
        )
    return completed.stdout


def _git_bytes(
    runner: GitRunner,
    arguments: Sequence[str],
    root: Path,
    *,
    label: str,
) -> bytes:
    value = runner(tuple(arguments), root)
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"git runner returned non-bytes for {label}")
    return bytes(value)


def _git_text(
    runner: GitRunner,
    arguments: Sequence[str],
    root: Path,
    *,
    label: str,
) -> str:
    payload = _git_bytes(runner, arguments, root, label=label)
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"git output for {label} is not UTF-8") from error


def _verify_repository_identity(root: Path, runner: GitRunner) -> tuple[str, str]:
    observed_root = Path(
        _git_text(
            runner,
            ("rev-parse", "--show-toplevel"),
            root,
            label="repository root",
        )
    ).resolve()
    if observed_root != root:
        raise ValueError("formal repository root drifted")
    branch = _git_text(
        runner, ("branch", "--show-current"), root, label="current branch"
    )
    if branch != EXPECTED_BRANCH:
        raise ValueError("formal handoff is on the wrong Git branch")
    content_commit = _checked_commit(
        _git_text(runner, ("rev-parse", "HEAD"), root, label="HEAD"),
        label="content commit",
    )
    remote_url = _git_text(
        runner,
        ("remote", "get-url", EXPECTED_REMOTE),
        root,
        label="configured remote",
    )
    if not remote_url:
        raise ValueError("formal Git remote has an empty URL")
    return branch, content_commit


def _verify_successful_remote(
    root: Path,
    runner: GitRunner,
    *,
    content_commit: str,
) -> None:
    observed = _git_text(
        runner,
        ("ls-remote", "--heads", EXPECTED_REMOTE, EXPECTED_REMOTE_REF),
        root,
        label="substantive remote ref",
    )
    if observed != f"{content_commit}\t{EXPECTED_REMOTE_REF}":
        raise ValueError("substantive remote ref does not equal the content commit")


def sanitize_push_error(value: str) -> str:
    """Return a bounded diagnostic safe for a public JSON/report artifact."""

    if not isinstance(value, str):
        raise TypeError("push error must be text")
    cleaned = _ANSI_ESCAPE.sub("", value)
    cleaned = _URL.sub("<url>", cleaned)
    cleaned = _TOKEN.sub("<credential>", cleaned)
    cleaned = _AUTH.sub("<credential>", cleaned)
    cleaned = _WINDOWS_PATH.sub("<path>", cleaned)
    cleaned = _POSIX_PATH.sub("<path>", cleaned)
    cleaned = "".join(
        character if ord(character) >= 32 else " " for character in cleaned
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        cleaned = "git push failed; no diagnostic was captured"
    if len(cleaned) > MAXIMUM_SANITIZED_ERROR_CHARACTERS:
        suffix = " … [truncated]"
        cleaned = (
            cleaned[: MAXIMUM_SANITIZED_ERROR_CHARACTERS - len(suffix)].rstrip()
            + suffix
        )
    _validate_sanitized_push_error(cleaned)
    return cleaned


def _validate_sanitized_push_error(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("failed push requires a nonempty sanitized error")
    if len(value) > MAXIMUM_SANITIZED_ERROR_CHARACTERS:
        raise ValueError("sanitized push error is too long")
    if value != " ".join(value.split()) or any(
        ord(character) < 32 for character in value
    ):
        raise ValueError(
            "sanitized push error contains control or noncanonical whitespace"
        )
    if _URL.search(value) or _FORBIDDEN_ERROR.search(value):
        raise ValueError("push error is not safely sanitized")
    return value


def _artifact_hashes_at_content_commit(
    root: Path,
    runner: GitRunner,
    *,
    content_commit: str,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in TRACKED_CONTENT_PATHS:
        working = _read_public_bytes(root, relative, label=relative)
        committed = _git_bytes(
            runner,
            ("show", f"{content_commit}:{relative}"),
            root,
            label=f"committed bytes for {relative}",
        )
        if committed != working:
            raise ValueError(
                f"working bytes differ from the content commit: {relative}"
            )
        hashes[relative] = _sha256(working)
    return hashes


def _atomic_exclusive_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"public commit marker already exists: {path.name}")
    encoded = _json_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_git_handoff(
    *,
    substantive_push_status: str,
    push_error: str | None = None,
    repository_root: Path | str = REPOSITORY_ROOT,
    git_runner: GitRunner | None = None,
) -> Mapping[str, Any]:
    """Create the exact post-push manifest without reading outcome artifacts.

    The caller performs the substantive ``git push`` first.  A successful
    status is accepted only if the current remote ref resolves to ``HEAD``.  A
    failed status never invents a remote SHA and records a sanitized version
    of the caller-supplied Git diagnostic.
    """

    if substantive_push_status not in PUSH_STATUSES:
        raise ValueError("substantive push status is invalid")
    root = _resolved_root(repository_root)
    runner = _default_git_runner if git_runner is None else git_runner

    reporting_bytes = _read_public_bytes(
        root, REPORTING_MARKER_PATH, label="reporting provenance"
    )
    reporting = _parse_json(reporting_bytes, label="reporting provenance")
    report_relative = SCIENTIFIC_PUBLIC_ARTIFACTS["final_report"]
    source_report = _read_public_bytes(root, report_relative, label="final report")
    if source_report.count(HANDOFF_MARKER.encode("utf-8")) != 1:
        raise ValueError(
            "scientific report must contain exactly one Git handoff marker"
        )
    public_hashes = _validate_reporting_marker(reporting, source_report=source_report)
    for role, relative in SCIENTIFIC_PUBLIC_ARTIFACTS.items():
        observed = (
            source_report
            if role == "final_report"
            else _read_public_bytes(root, relative, label=role)
        )
        if _sha256(observed) != public_hashes[role]:
            raise ValueError(f"public reporting artifact drifted: {role}")
    evaluation_lock_bytes = _read_public_bytes(
        root, EVALUATION_LOCK_PATH, label="EVALUATION_LOCK"
    )
    _validate_frozen_handoff_sources(
        root,
        reporting=reporting,
        evaluation_lock_bytes=evaluation_lock_bytes,
    )

    branch, content_commit = _verify_repository_identity(root, runner)

    if substantive_push_status == PUSH_OK:
        if push_error is not None:
            raise ValueError("successful push must not have a push error")
        _verify_successful_remote(root, runner, content_commit=content_commit)
        remote_sha: str | None = content_commit
        sanitized_error: str | None = None
    else:
        if push_error is None:
            raise ValueError("failed push requires the captured Git diagnostic")
        remote_sha = None
        sanitized_error = sanitize_push_error(push_error)

    artifacts = _artifact_hashes_at_content_commit(
        root, runner, content_commit=content_commit
    )
    manifest: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "content_commit_sha": content_commit,
        "branch": branch,
        "remote": EXPECTED_REMOTE,
        "remote_ref": EXPECTED_REMOTE_REF,
        "substantive_push_status": substantive_push_status,
        "substantive_remote_ref_sha": remote_sha,
        "sanitized_push_error": sanitized_error,
        "artifact_sha256": artifacts,
    }
    destination = _fixed_path(root, GIT_HANDOFF_PATH)
    _atomic_exclusive_public_json(destination, manifest)
    return manifest


def _validate_digest_mapping(
    value: Any,
    expected_keys: set[str],
    *,
    label: str,
) -> Mapping[str, str]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{label} exact key set drifted")
    return {
        str(key): _checked_digest(digest, label=f"{label} {key}")
        for key, digest in value.items()
    }


def _validate_manifest_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != _HANDOFF_KEYS:
        raise ValueError("git handoff manifest exact schema drifted")
    if payload["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ValueError("git handoff schema version drifted")
    content_commit = _checked_commit(
        payload["content_commit_sha"], label="git handoff content commit"
    )
    expected_fixed = {
        "branch": EXPECTED_BRANCH,
        "remote": EXPECTED_REMOTE,
        "remote_ref": EXPECTED_REMOTE_REF,
    }
    for key, expected in expected_fixed.items():
        if payload[key] != expected:
            raise ValueError(f"git handoff {key} drifted")
    status = payload["substantive_push_status"]
    if status not in PUSH_STATUSES:
        raise ValueError("git handoff push status is invalid")
    if status == PUSH_OK:
        if payload["sanitized_push_error"] is not None:
            raise ValueError("successful push must have a null error")
        if payload["substantive_remote_ref_sha"] != content_commit:
            raise ValueError("successful push remote SHA differs from content commit")
    else:
        if payload["substantive_remote_ref_sha"] is not None:
            raise ValueError("failed push must not claim a remote SHA")
        _validate_sanitized_push_error(payload["sanitized_push_error"])
    _validate_digest_mapping(
        payload["artifact_sha256"],
        set(TRACKED_CONTENT_PATHS),
        label="git handoff artifact hashes",
    )
    return payload


def _validate_reporting_marker(
    payload: Mapping[str, Any],
    *,
    source_report: bytes,
) -> Mapping[str, str]:
    if set(payload) != _REPORTING_KEYS:
        raise ValueError("reporting provenance exact schema drifted")
    if payload["schema_version"] != REPORTING_SCHEMA_VERSION:
        raise ValueError("reporting provenance schema version drifted")
    if payload["model_name"] != EXPECTED_MODEL_NAME or payload["posthoc"] is not True:
        raise ValueError("reporting provenance model/post-hoc identity drifted")
    if payload["published_last"] is not True or payload["lineage_mode"] != "formal":
        raise ValueError("formal reporting provenance was not published last")
    if not isinstance(payload["report_lock"], dict) or not isinstance(
        payload["producers"], dict
    ):
        raise ValueError("formal reporting lineage is incomplete")
    _checked_digest(
        payload["comparison_contract_sha256"], label="comparison contract hash"
    )
    public_hashes = _validate_digest_mapping(
        payload["public_artifact_sha256"],
        set(SCIENTIFIC_PUBLIC_ARTIFACTS),
        label="reporting public artifact hashes",
    )
    if public_hashes["final_report"] != _sha256(source_report):
        raise ValueError("scientific report bytes differ from the reporting marker")
    return public_hashes


def _validate_frozen_handoff_sources(
    root: Path,
    *,
    reporting: Mapping[str, Any],
    evaluation_lock_bytes: bytes,
) -> None:
    """Close the outcome-blind lock chain without following runtime inputs."""

    report_lock = reporting.get("report_lock")
    if not isinstance(report_lock, dict) or set(report_lock) != {
        "lock_sha256",
        "argv_sha256",
    }:
        raise ValueError("reporting provenance report-lock schema drifted")
    expected_lock_sha = _checked_digest(
        report_lock["lock_sha256"], label="reporting evaluation-lock hash"
    )
    _checked_digest(report_lock["argv_sha256"], label="report argv hash")
    if _sha256(evaluation_lock_bytes) != expected_lock_sha:
        raise ValueError("EVALUATION_LOCK bytes differ from reporting provenance")

    evaluation_path = _fixed_path(root, EVALUATION_LOCK_PATH)
    if evaluation_path.lstat().st_mode & 0o222:
        raise PermissionError("formal EVALUATION_LOCK must remain read-only")
    evaluation = _parse_json(evaluation_lock_bytes, label="EVALUATION_LOCK")
    if set(evaluation) != _EVALUATION_LOCK_KEYS:
        raise ValueError("EVALUATION_LOCK exact schema drifted")
    if evaluation["schema_version"] != "foundation_mri_dinov3_evaluation_lock_v1":
        raise ValueError("EVALUATION_LOCK schema version drifted")
    if evaluation["status"] != "FROZEN_BEFORE_DINOV3_OUTCOME_EVALUATION":
        raise ValueError("EVALUATION_LOCK was not frozen before outcome evaluation")
    visibility = evaluation["prior_visibility"]
    expected_visibility = {
        "original_study_outcomes_public": True,
        "extension_is_post_hoc": True,
        "no_preregistration_claim": True,
        "dinov3_outcome_metrics_seen": False,
    }
    if not isinstance(visibility, dict) or any(
        visibility.get(key) is not expected
        for key, expected in expected_visibility.items()
    ):
        raise ValueError("EVALUATION_LOCK visibility gate drifted")

    # Deliberately do not resolve or read runtime_inputs, feature_asset, parent
    # comparators, or any other outcome-adjacent record here.  Only the four
    # finalization sources that must have been frozen outcome-blind are opened.
    locked_files = evaluation["locked_files"]
    if not isinstance(locked_files, dict):
        raise ValueError("EVALUATION_LOCK locked_files must be an object")
    missing = set(FROZEN_HANDOFF_SOURCES) - set(locked_files)
    if missing:
        raise ValueError(
            f"EVALUATION_LOCK lacks frozen handoff roles: {sorted(missing)}"
        )
    for role, relative in FROZEN_HANDOFF_SOURCES.items():
        record = locked_files[role]
        if not isinstance(record, dict) or set(record) not in (
            {"path", "sha256"},
            {"path", "sha256", "bytes"},
        ):
            raise ValueError(f"invalid frozen handoff record schema: {role}")
        if record["path"] != relative:
            raise ValueError(f"frozen handoff path drifted: {role}")
        payload = _read_public_bytes(root, relative, label=f"frozen {role}")
        if "bytes" in record:
            size = record["bytes"]
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size != len(payload)
            ):
                raise ValueError(f"frozen handoff byte size drifted: {role}")
        expected = _checked_digest(record["sha256"], label=f"frozen {role}")
        if _sha256(payload) != expected:
            raise ValueError(f"frozen handoff SHA-256 drifted: {role}")


def _markdown_cell(value: Any) -> str:
    text = " ".join(str(value).split())
    return html.escape(text, quote=True).replace("|", "\\|")


def render_git_handoff_section(
    handoff: Mapping[str, Any],
    *,
    handoff_sha256: str,
) -> str:
    """Render the fixed, metric-free Git metadata section."""

    validated = _validate_manifest_schema(handoff)
    manifest_sha = _checked_digest(handoff_sha256, label="git handoff manifest")
    remote_sha = validated["substantive_remote_ref_sha"]
    error = validated["sanitized_push_error"]
    rows = (
        ("Content commit", validated["content_commit_sha"]),
        ("Branch", validated["branch"]),
        ("Attempted remote/ref", f"{validated['remote']} {validated['remote_ref']}"),
        ("substantive_push_status", validated["substantive_push_status"]),
        ("substantive_remote_ref_sha", "未建立" if remote_sha is None else remote_sha),
        ("Sanitized push error", "无" if error is None else error),
        (
            "Scientific source report SHA-256",
            validated["artifact_sha256"][SCIENTIFIC_PUBLIC_ARTIFACTS["final_report"]],
        ),
        ("Git handoff manifest SHA-256", manifest_sha),
    )
    lines = [
        "## Git 交接（后置元数据）",
        "",
        "本段由 outcome-blind finalizer 在 scientific report 的 substantive content commit 后加入；"
        "它不读取 patient-level/private 输入，也不改变上方科学内容。",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    lines.extend(
        f"| {_markdown_cell(key)} | {_markdown_cell(value)} |" for key, value in rows
    )
    return "\n".join(lines)


def _source_and_final_report(
    observed: bytes,
    *,
    section: str,
) -> tuple[bytes, bytes, bool]:
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("final report is not UTF-8") from error
    marker_count = text.count(HANDOFF_MARKER)
    section_count = text.count(section)
    if marker_count == 1 and section_count == 0:
        source = observed
        final_text = text.replace(HANDOFF_MARKER, section, 1)
        return source, final_text.encode("utf-8"), False
    if marker_count == 0 and section_count == 1:
        source_text = text.replace(section, HANDOFF_MARKER, 1)
        final = observed
        return source_text.encode("utf-8"), final, True
    raise ValueError(
        "final report has neither one source marker nor one recoverable Git section"
    )


def _validate_git_and_content(
    root: Path,
    runner: GitRunner,
    *,
    handoff: Mapping[str, Any],
    source_report: bytes,
    reporting_marker: bytes,
) -> None:
    branch, head = _verify_repository_identity(root, runner)
    if branch != handoff["branch"] or head != handoff["content_commit_sha"]:
        raise ValueError("Git HEAD/branch differs from the handoff manifest")
    if handoff["substantive_push_status"] == PUSH_OK:
        _verify_successful_remote(root, runner, content_commit=head)

    hashes = handoff["artifact_sha256"]
    for relative in TRACKED_CONTENT_PATHS:
        if relative == SCIENTIFIC_PUBLIC_ARTIFACTS["final_report"]:
            working = source_report
        elif relative == REPORTING_MARKER_PATH:
            working = reporting_marker
        else:
            working = _read_public_bytes(root, relative, label=relative)
        if _sha256(working) != hashes[relative]:
            raise ValueError(f"git handoff artifact hash drifted: {relative}")
        committed = _git_bytes(
            runner,
            ("show", f"{head}:{relative}"),
            root,
            label=f"committed bytes for {relative}",
        )
        if committed != working:
            raise ValueError(f"content commit bytes drifted: {relative}")


def _coverage_payload(
    *,
    handoff: Mapping[str, Any],
    source_report: bytes,
    final_report: bytes,
    handoff_bytes: bytes,
    reporting_marker_bytes: bytes,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "publication_commit_marker": "coverage_receipt",
        "source_report_sha256": _sha256(source_report),
        "final_report_sha256": _sha256(final_report),
        "git_handoff_sha256": _sha256(handoff_bytes),
        "reporting_run_provenance_sha256": _sha256(reporting_marker_bytes),
        "content_commit_sha": handoff["content_commit_sha"],
        "branch": handoff["branch"],
        "remote": handoff["remote"],
        "remote_ref": handoff["remote_ref"],
        "substantive_push_status": handoff["substantive_push_status"],
        "substantive_remote_ref_sha": handoff["substantive_remote_ref_sha"],
        "private_inputs_read": False,
    }
    if set(payload) != _COVERAGE_KEYS:
        raise AssertionError("coverage receipt schema construction drifted")
    return payload


def _atomic_replace_public(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
        os.chmod(path, 0o644)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize_handoff(
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
    git_runner: GitRunner | None = None,
) -> Mapping[str, Any]:
    """Attach Git metadata and publish coverage last, without outcome reads.

    If a prior attempt replaced the report but failed before publishing the
    coverage receipt, the exact Git section is reversed to reconstruct and
    revalidate the source report before publication resumes.  Once coverage
    exists, a second invocation is rejected.
    """

    root = _resolved_root(repository_root)
    runner = _default_git_runner if git_runner is None else git_runner
    coverage_path = _fixed_path(root, COVERAGE_PATH)
    if os.path.lexists(coverage_path):
        raise FileExistsError("final report handoff is already finalized")

    handoff_bytes = _read_public_bytes(root, GIT_HANDOFF_PATH, label="git handoff")
    handoff = _validate_manifest_schema(
        _parse_json(handoff_bytes, label="git handoff manifest")
    )
    reporting_bytes = _read_public_bytes(
        root, REPORTING_MARKER_PATH, label="reporting provenance"
    )
    reporting = _parse_json(reporting_bytes, label="reporting provenance")
    report_relative = SCIENTIFIC_PUBLIC_ARTIFACTS["final_report"]
    observed_report = _read_public_bytes(root, report_relative, label="final report")
    section = render_git_handoff_section(handoff, handoff_sha256=_sha256(handoff_bytes))
    source_report, final_report, recovering = _source_and_final_report(
        observed_report, section=section
    )
    public_hashes = _validate_reporting_marker(reporting, source_report=source_report)
    evaluation_lock_bytes = _read_public_bytes(
        root, EVALUATION_LOCK_PATH, label="EVALUATION_LOCK"
    )
    _validate_frozen_handoff_sources(
        root,
        reporting=reporting,
        evaluation_lock_bytes=evaluation_lock_bytes,
    )
    for role, relative in SCIENTIFIC_PUBLIC_ARTIFACTS.items():
        if role == "final_report":
            observed = source_report
        else:
            observed = _read_public_bytes(root, relative, label=role)
        if _sha256(observed) != public_hashes[role]:
            raise ValueError(f"public reporting artifact drifted: {role}")

    _validate_git_and_content(
        root,
        runner,
        handoff=handoff,
        source_report=source_report,
        reporting_marker=reporting_bytes,
    )
    receipt = _coverage_payload(
        handoff=handoff,
        source_report=source_report,
        final_report=final_report,
        handoff_bytes=handoff_bytes,
        reporting_marker_bytes=reporting_bytes,
    )

    report_path = _fixed_path(root, report_relative)
    if recovering:
        if observed_report != final_report:
            raise AssertionError("recoverable final report bytes drifted")
    else:
        # Recheck the source immediately before replacement.  Publication is
        # recoverable if the subsequent exclusive receipt link fails.
        if (
            _read_public_bytes(root, report_relative, label="final report")
            != source_report
        ):
            raise ValueError("scientific report changed during finalization")
        _atomic_replace_public(report_path, final_report)
    if _read_public_bytes(root, report_relative, label="final report") != final_report:
        raise ValueError("augmented final report publication failed verification")
    _atomic_exclusive_public_json(coverage_path, receipt)
    return {
        "status": "complete",
        "recovered": recovering,
        "private_inputs_read": False,
        "source_report_sha256": receipt["source_report_sha256"],
        "final_report_sha256": receipt["final_report_sha256"],
        "git_handoff_sha256": receipt["git_handoff_sha256"],
    }


__all__ = [
    "COVERAGE_PATH",
    "COVERAGE_SCHEMA_VERSION",
    "EVALUATION_LOCK_PATH",
    "EXPECTED_BRANCH",
    "EXPECTED_REMOTE",
    "EXPECTED_REMOTE_REF",
    "FROZEN_HANDOFF_SOURCES",
    "GIT_HANDOFF_PATH",
    "HANDOFF_MARKER",
    "HANDOFF_SCHEMA_VERSION",
    "PUSH_FAILED",
    "PUSH_OK",
    "REPORTING_MARKER_PATH",
    "SCIENTIFIC_PUBLIC_ARTIFACTS",
    "TRACKED_CONTENT_PATHS",
    "create_git_handoff",
    "finalize_handoff",
    "render_git_handoff_section",
    "sanitize_push_error",
]
