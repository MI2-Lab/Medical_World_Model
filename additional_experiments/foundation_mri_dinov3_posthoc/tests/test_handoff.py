from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from foundation_mri_dinov3 import handoff as hm


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = EXPERIMENT_ROOT / "scripts/finalize_handoff.py"
TEMPLATE_PATH = EXPERIMENT_ROOT / "reports/final_report.template.md"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _write(root: Path, relative: str, payload: bytes) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise AssertionError(
            f"synthetic git command failed: {arguments!r}: "
            f"{completed.stderr.decode('utf-8', errors='replace')}"
        )
    return completed.stdout


@dataclass(frozen=True)
class SyntheticBundle:
    root: Path
    remote: Path
    source_report: bytes
    content_commit: str

    @property
    def report_path(self) -> Path:
        return self.root / hm.SCIENTIFIC_PUBLIC_ARTIFACTS["final_report"]

    @property
    def marker_path(self) -> Path:
        return self.root / hm.REPORTING_MARKER_PATH

    @property
    def handoff_path(self) -> Path:
        return self.root / hm.GIT_HANDOFF_PATH

    @property
    def coverage_path(self) -> Path:
        return self.root / hm.COVERAGE_PATH


def _make_bundle(tmp_path: Path, *, pushed: bool) -> SyntheticBundle:
    root = tmp_path / "repository"
    remote = tmp_path / "remote.git"
    root.mkdir(parents=True)
    _git(root, "init", "--initial-branch", hm.EXPECTED_BRANCH)
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")

    source_report = (
        "# Synthetic scientific report\n\n"
        "No patient data or formal outcome is present.\n\n"
        f"{hm.HANDOFF_MARKER}\n"
    ).encode("utf-8")
    public_payloads = {
        "paired_comparisons": b"comparison_id,metric\nsynthetic,auroc\n",
        "results_summary": b'{"synthetic":true}\n',
        "final_report": source_report,
        "timing_figure": b"synthetic-png-one",
        "comparison_figure": b"synthetic-png-two",
    }
    for role, relative in hm.SCIENTIFIC_PUBLIC_ARTIFACTS.items():
        _write(root, relative, public_payloads[role])

    # The finalizer verifies that the implementation interpreting the manifest
    # was itself part of the substantive commit.  Synthetic repositories copy
    # only source bytes; no formal result file is read.
    implementation_payloads = {
        hm.FROZEN_HANDOFF_SOURCES["final_report_template"]: TEMPLATE_PATH.read_bytes(),
        hm.FROZEN_HANDOFF_SOURCES["handoff_module"]: Path(hm.__file__).read_bytes(),
        hm.FROZEN_HANDOFF_SOURCES["finalize_handoff_cli"]: CLI_PATH.read_bytes(),
        hm.FROZEN_HANDOFF_SOURCES["handoff_test"]: Path(__file__).read_bytes(),
    }
    for relative, payload in implementation_payloads.items():
        _write(root, relative, payload)

    locked_files = {
        role: {
            "path": relative,
            "sha256": _sha(implementation_payloads[relative]),
            "bytes": len(implementation_payloads[relative]),
        }
        for role, relative in hm.FROZEN_HANDOFF_SOURCES.items()
    }
    evaluation_lock = {
        "schema_version": "foundation_mri_dinov3_evaluation_lock_v1",
        "status": "FROZEN_BEFORE_DINOV3_OUTCOME_EVALUATION",
        "created_utc": "2026-01-01T00:00:00Z",
        "prior_visibility": {
            "original_study_outcomes_public": True,
            "extension_is_post_hoc": True,
            "no_preregistration_claim": True,
            "dinov3_outcome_metrics_seen": False,
        },
        "parent_model_input_lock": {},
        "parent_publication": {},
        "protocols": {},
        "locked_files": locked_files,
        # Deliberately nonexistent outcome-adjacent paths prove that handoff
        # finalization does not follow these records.
        "runtime_inputs": {
            "not_read": {
                "path": "/definitely/missing/synthetic.private.csv",
                "sha256": "7" * 64,
            }
        },
        "feature_asset": {
            "path": "/definitely/missing/features.private.npz",
            "sha256": "8" * 64,
        },
        "parent_comparator_artifacts": {},
        "commands": {"baseline": [], "probe": [], "report": []},
        "expected_counts": {"synthetic": 1},
        "exclusive_outputs": {},
    }
    evaluation_path = _write(
        root, hm.EVALUATION_LOCK_PATH, _json_bytes(evaluation_lock)
    )
    evaluation_path.chmod(0o444)

    artifact_hashes = {
        role: _sha(public_payloads[role]) for role in hm.SCIENTIFIC_PUBLIC_ARTIFACTS
    }
    reporting_marker = {
        "schema_version": hm.REPORTING_SCHEMA_VERSION,
        "summary_schema_version": "synthetic_summary_v1",
        "model_name": hm.EXPECTED_MODEL_NAME,
        "posthoc": True,
        "comparison_contract_sha256": "1" * 64,
        "comparison_spec_count": 84,
        "paired_metric_row_count": 252,
        "input_sha256": {"synthetic_public_input": "2" * 64},
        "public_artifact_sha256": artifact_hashes,
        "published_last": True,
        "lineage_mode": "formal",
        "report_lock": {
            "lock_sha256": _sha(evaluation_path.read_bytes()),
            "argv_sha256": "4" * 64,
        },
        "producers": {
            "baseline": {"receipt_sha256": "5" * 64},
            "probe": {"receipt_sha256": "6" * 64},
            "parent_comparator": {"required_csv_count": 5},
        },
    }
    _write(root, hm.REPORTING_MARKER_PATH, _json_bytes(reporting_marker))

    _git(root, "add", "--", *hm.TRACKED_CONTENT_PATHS)
    _git(root, "commit", "-m", "synthetic substantive content")
    content_commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if pushed:
        _git(tmp_path, "init", "--bare", str(remote))
        _git(root, "remote", "add", hm.EXPECTED_REMOTE, str(remote))
        _git(root, "push", "-u", hm.EXPECTED_REMOTE, hm.EXPECTED_BRANCH)
    else:
        _git(root, "remote", "add", hm.EXPECTED_REMOTE, str(tmp_path / "missing.git"))
    return SyntheticBundle(root, remote, source_report, content_commit)


def _create_success_manifest(bundle: SyntheticBundle) -> dict[str, Any]:
    value = hm.create_git_handoff(
        substantive_push_status=hm.PUSH_OK,
        repository_root=bundle.root,
    )
    assert isinstance(value, dict)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_template_contains_one_fixed_non_placeholder_marker() -> None:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert text.count(hm.HANDOFF_MARKER) == 1
    assert "{{GIT" not in text


def test_success_manifest_and_finalization_bind_both_report_versions(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path, pushed=True)
    manifest = _create_success_manifest(bundle)
    assert set(manifest) == {
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
    assert manifest["content_commit_sha"] == bundle.content_commit
    assert manifest["substantive_remote_ref_sha"] == bundle.content_commit
    assert manifest["sanitized_push_error"] is None
    assert set(manifest["artifact_sha256"]) == set(hm.TRACKED_CONTENT_PATHS)

    result = hm.finalize_handoff(repository_root=bundle.root)
    assert result["status"] == "complete"
    assert result["recovered"] is False
    assert result["private_inputs_read"] is False
    final_report = bundle.report_path.read_bytes()
    assert hm.HANDOFF_MARKER.encode() not in final_report
    assert hm.PUSH_OK.encode() in final_report
    assert hm.EXPECTED_BRANCH.encode() in final_report
    assert bundle.content_commit.encode() in final_report

    coverage = _load_json(bundle.coverage_path)
    assert set(coverage) == {
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
    assert coverage["schema_version"] == hm.COVERAGE_SCHEMA_VERSION
    assert coverage["publication_commit_marker"] == "coverage_receipt"
    assert coverage["source_report_sha256"] == _sha(bundle.source_report)
    assert coverage["final_report_sha256"] == _sha(final_report)
    assert coverage["git_handoff_sha256"] == _sha(bundle.handoff_path.read_bytes())
    assert coverage["reporting_run_provenance_sha256"] == _sha(
        bundle.marker_path.read_bytes()
    )
    assert coverage["private_inputs_read"] is False
    assert (
        _git(
            bundle.root,
            "show",
            f"{bundle.content_commit}:{hm.SCIENTIFIC_PUBLIC_ARTIFACTS['final_report']}",
        )
        == bundle.source_report
    )

    with pytest.raises(FileExistsError, match="already finalized"):
        hm.finalize_handoff(repository_root=bundle.root)


def test_failed_push_is_sanitized_and_never_claims_a_remote_sha(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path, pushed=False)
    raw = (
        "fatal: https://user:password@github.example/repo ghp_SUPERSECRET "
        "token=also-secret /home/alice/private/repository\nnetwork unreachable"
    )
    manifest = hm.create_git_handoff(
        substantive_push_status=hm.PUSH_FAILED,
        push_error=raw,
        repository_root=bundle.root,
    )
    error = str(manifest["sanitized_push_error"])
    assert manifest["substantive_remote_ref_sha"] is None
    assert "SUPERSECRET" not in error
    assert "also-secret" not in error
    assert "/home/" not in error
    assert "https://" not in error
    assert "\n" not in error

    # The configured remote does not exist.  Passing finalization demonstrates
    # that failed status validates local truth without fabricating/looking up a
    # successful remote SHA.
    result = hm.finalize_handoff(repository_root=bundle.root)
    assert result["status"] == "complete"
    report = bundle.report_path.read_text(encoding="utf-8")
    assert hm.PUSH_FAILED in report
    assert "未建立" in report
    assert html.escape(error, quote=True).replace("|", "\\|") in report


def test_recovery_finishes_when_report_was_replaced_before_coverage(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path, pushed=True)
    manifest = _create_success_manifest(bundle)
    handoff_bytes = bundle.handoff_path.read_bytes()
    section = hm.render_git_handoff_section(
        manifest, handoff_sha256=_sha(handoff_bytes)
    )
    bundle.report_path.write_text(
        bundle.source_report.decode("utf-8").replace(hm.HANDOFF_MARKER, section),
        encoding="utf-8",
    )
    assert not bundle.coverage_path.exists()

    result = hm.finalize_handoff(repository_root=bundle.root)
    assert result["recovered"] is True
    assert bundle.coverage_path.is_file()
    assert _load_json(bundle.coverage_path)["private_inputs_read"] is False


def test_manifest_exact_schema_and_sanitized_error_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path, pushed=False)
    hm.create_git_handoff(
        substantive_push_status=hm.PUSH_FAILED,
        push_error="synthetic transport failure",
        repository_root=bundle.root,
    )
    manifest = _load_json(bundle.handoff_path)
    manifest["unexpected"] = True
    bundle.handoff_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(ValueError, match="exact schema"):
        hm.finalize_handoff(repository_root=bundle.root)

    bundle.handoff_path.unlink()
    hm.create_git_handoff(
        substantive_push_status=hm.PUSH_FAILED,
        push_error="synthetic transport failure",
        repository_root=bundle.root,
    )
    manifest = _load_json(bundle.handoff_path)
    manifest["sanitized_push_error"] = "password=not-sanitized /home/alice"
    bundle.handoff_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(ValueError, match="safely sanitized"):
        hm.finalize_handoff(repository_root=bundle.root)


def test_public_artifact_and_reporting_source_hash_drift_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "artifact", pushed=True)
    _create_success_manifest(bundle)
    paired = bundle.root / hm.SCIENTIFIC_PUBLIC_ARTIFACTS["paired_comparisons"]
    paired.write_bytes(paired.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="public reporting artifact drifted"):
        hm.finalize_handoff(repository_root=bundle.root)

    second = _make_bundle(tmp_path / "report", pushed=True)
    _create_success_manifest(second)
    marker = _load_json(second.marker_path)
    marker["public_artifact_sha256"]["final_report"] = "0" * 64
    second.marker_path.write_bytes(_json_bytes(marker))
    with pytest.raises(ValueError, match="scientific report bytes"):
        hm.finalize_handoff(repository_root=second.root)


def test_success_status_rechecks_remote_ref(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, pushed=True)
    _create_success_manifest(bundle)
    _git(
        bundle.root,
        "--git-dir",
        str(bundle.remote),
        "update-ref",
        "-d",
        hm.EXPECTED_REMOTE_REF,
    )
    with pytest.raises(ValueError, match="remote ref"):
        hm.finalize_handoff(repository_root=bundle.root)


def test_handoff_creation_rejects_uncommitted_tracked_bytes_and_overwrite(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "drift", pushed=True)
    template = bundle.root / f"{hm.EXPERIMENT_PREFIX}/reports/final_report.template.md"
    template.write_bytes(template.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="frozen handoff byte size drifted"):
        hm.create_git_handoff(
            substantive_push_status=hm.PUSH_OK,
            repository_root=bundle.root,
        )

    second = _make_bundle(tmp_path / "overwrite", pushed=True)
    _create_success_manifest(second)
    with pytest.raises(FileExistsError, match="already exists"):
        hm.create_git_handoff(
            substantive_push_status=hm.PUSH_OK,
            repository_root=second.root,
        )


def test_outcome_blind_evaluation_lock_prevents_postfreeze_source_attack(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "source-attack", pushed=True)
    module_path = bundle.root / hm.FROZEN_HANDOFF_SOURCES["handoff_module"]
    module_path.write_bytes(module_path.read_bytes() + b"\n# post-freeze attack\n")
    _git(bundle.root, "add", "--", hm.FROZEN_HANDOFF_SOURCES["handoff_module"])
    _git(bundle.root, "commit", "-m", "synthetic post-freeze source attack")
    _git(bundle.root, "push", hm.EXPECTED_REMOTE, hm.EXPECTED_BRANCH)
    with pytest.raises(ValueError, match="frozen handoff byte size drifted"):
        hm.create_git_handoff(
            substantive_push_status=hm.PUSH_OK,
            repository_root=bundle.root,
        )

    marker_attack = _make_bundle(tmp_path / "marker-attack", pushed=True)
    marker = _load_json(marker_attack.marker_path)
    marker["report_lock"]["lock_sha256"] = "0" * 64
    marker_attack.marker_path.write_bytes(_json_bytes(marker))
    with pytest.raises(ValueError, match="EVALUATION_LOCK bytes"):
        hm.create_git_handoff(
            substantive_push_status=hm.PUSH_OK,
            repository_root=marker_attack.root,
        )


def test_finalizer_requires_read_only_evaluation_lock(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, pushed=True)
    _create_success_manifest(bundle)
    (bundle.root / hm.EVALUATION_LOCK_PATH).chmod(0o644)
    with pytest.raises(PermissionError, match="must remain read-only"):
        hm.finalize_handoff(repository_root=bundle.root)
    assert bundle.report_path.read_bytes() == bundle.source_report
    assert not bundle.coverage_path.exists()


def test_cli_rejects_nonempty_argv_before_formal_access() -> None:
    spec = importlib.util.spec_from_file_location(
        "synthetic_finalize_handoff_cli", CLI_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="exact empty argv"):
        module.main(["--not-formal"])
