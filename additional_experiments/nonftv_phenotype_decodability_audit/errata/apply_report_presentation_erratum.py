#!/usr/bin/env python3
"""Apply and audit a post-lock, presentation-only report correction.

The preregistered analyzer is intentionally left byte-for-byte unchanged.  Its
``_metric_sentence`` helper computes a label but omits the intended return,
which caused ten descriptive summaries in ``final_report.md`` to render as a
missing value.  This delivery utility:

1. fully verifies the immutable preregistration lock and frozen analyzer;
2. snapshots every non-report file in the experiment (hash, size, and mode);
3. replaces only ``_metric_sentence`` in the imported module object;
4. invokes the locked analyzer's aggregate-only ``--report-only`` entry point;
5. appends a transparent erratum notice and verifies the ten corrected lines;
6. proves the entire non-report snapshot is unchanged; and
7. writes an immutable, no-replace JSON audit manifest.

This file was added after the lock.  It is deliberately located under
``errata/`` rather than ``scripts/`` and must never be described as
preregistered scientific analysis code.  It does not open or parse the private
OOF predictions; their compressed bytes are hashed only as an invariant.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_ROOT.parents[1]
SCRIPTS_ROOT = AUDIT_ROOT / "scripts"
REPORT_PATH = AUDIT_ROOT / "reports" / "final_report.md"
LOCK_PATH = AUDIT_ROOT / "PREREGISTRATION_LOCK.json"
ANALYZER_PATH = SCRIPTS_ROOT / "analyze_results.py"
RUN_SUMMARY_PATH = AUDIT_ROOT / "metrics" / "run_summary.json"
PROVENANCE_PATH = AUDIT_ROOT / "manifests" / "input_provenance.json"
BEST_CELLS_PATH = AUDIT_ROOT / "metrics" / "descriptive_best_cells.csv"
GATES_PATH = AUDIT_ROOT / "metrics" / "primary_gates.json"
SCORECARD_PATH = AUDIT_ROOT / "metrics" / "grounding_candidate_scorecard.csv"
RECOMMENDATION_PATH = AUDIT_ROOT / "metrics" / "final_target_recommendation.csv"
PRIVATE_OOF_PATH = AUDIT_ROOT / "predictions" / "oof_predictions.private.csv.gz"
ERRATA_ROOT = Path(__file__).resolve().parent
BASE_MANIFEST_PATH = ERRATA_ROOT / "report_presentation_erratum.json"

EXPECTED_LOCK_SHA256 = "b3e9809f47a13b2db2c958cee4bec112b18273de75606c05538bc2fc04f706ee"
EXPECTED_ANALYZER_SHA256 = "1326e48a0e03f276f98ac2ad3fb76f713d356724eb9fa226f6236e1b8b058750"
EXPECTED_PARENT_SHA = "7742d737d92ed153b5c721cd323528b0a127d5ef"
EXPECTED_BRANCH = "feature/nonftv-phenotype-decodability-audit"
EXPECTED_BINDING_COUNT = 133
EXPECTED_GATE_RESULTS = {"A": False, "B": False, "C": True, "D": True, "E": False}
EXPECTED_CLASSIFICATIONS = {
    "LD": "Class D — CURRENTLY NOT IMAGE-OBSERVABLE",
    "SPH": "MIXED OR UNRESOLVED",
    "BPE": "FOV-BLOCKED — A/D CLASSIFICATION NOT AUTHORIZED",
}
EXPECTED_RECOMMENDATION = "Need broader-context phenotype branch"
BUGGY_SUMMARY_COUNT = 10
BASE_MANIFEST_SCHEMA = "post_lock_report_presentation_erratum/v1"
DELIVERY_MANIFEST_SCHEMA = "post_lock_report_delivery_render/v1"
NOTICE_HEADING = "## 冻结后报告呈现勘误（presentation-only）"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(AUDIT_ROOT))


def _atomic_replace_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace immutable erratum manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _scientific_snapshot() -> dict[str, dict[str, Any]]:
    """Hash every existing experiment file except the report and errata chain."""

    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(AUDIT_ROOT.rglob("*")):
        relative = path.relative_to(AUDIT_ROOT)
        if relative == REPORT_PATH.relative_to(AUDIT_ROOT):
            continue
        if relative.parts and relative.parts[0] == "errata":
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden in scientific snapshot: {relative}")
        if not path.is_file():
            continue
        file_stat = path.stat()
        snapshot[str(relative)] = {
            "sha256": _file_sha256(path),
            "size_bytes": int(file_stat.st_size),
            "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
        }
    required = {
        "PREREGISTRATION_LOCK.json",
        "EXPERIMENT_PLAN.md",
        "configs/audit.json",
        "scripts/analyze_results.py",
        "scripts/audit_core.py",
        "scripts/freeze_preregistration.py",
        "scripts/run_audit.py",
        "scripts/validate_audit.py",
        "tests/test_audit_contracts.py",
        "metrics/run_summary.json",
        "manifests/input_provenance.json",
        "predictions/oof_predictions.private.csv.gz",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise FileNotFoundError(f"scientific snapshot is missing required files: {missing}")
    return snapshot


def _verify_static_contract(*, allow_descendant_head: bool) -> dict[str, Any]:
    if _file_sha256(LOCK_PATH) != EXPECTED_LOCK_SHA256:
        raise ValueError("immutable lock SHA-256 differs from the audited lock")
    if stat.S_IMODE(LOCK_PATH.stat().st_mode) != 0o444:
        raise PermissionError("immutable lock must retain mode 0444")
    if _file_sha256(ANALYZER_PATH) != EXPECTED_ANALYZER_SHA256:
        raise ValueError("locked analyzer bytes differ from the audited analyzer")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise ValueError("current Git branch differs from the locked audit branch")
    head = _git("rev-parse", "HEAD")
    if allow_descendant_head:
        status_code = subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_PARENT_SHA, head],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if status_code != 0:
            raise ValueError("current HEAD is not a descendant of the locked parent")
    elif head != EXPECTED_PARENT_SHA:
        raise ValueError("initial erratum render requires the exact locked parent HEAD")

    lock = _load_json(LOCK_PATH)
    analyzer_bindings = [
        binding
        for binding in lock["analysis_contract"]["files"]
        if binding.get("role") == "script:analyze_results.py"
    ]
    if len(analyzer_bindings) != 1:
        raise ValueError("lock must contain exactly one analyzer binding")
    binding = analyzer_bindings[0]
    if binding.get("sha256") != EXPECTED_ANALYZER_SHA256:
        raise ValueError("lock analyzer binding differs from the audited SHA-256")
    if Path(str(binding["path"])).resolve() != ANALYZER_PATH.resolve():
        candidate = (REPO_ROOT / str(binding["path"])).resolve()
        if candidate != ANALYZER_PATH.resolve():
            raise ValueError("lock analyzer binding resolves to an unexpected path")

    # Import only the frozen verifier after byte-level checks.  Disabling
    # bytecode writes keeps the non-report snapshot stable.
    sys.dont_write_bytecode = True
    scripts_value = str(SCRIPTS_ROOT)
    if scripts_value not in sys.path:
        sys.path.insert(0, scripts_value)
    from freeze_preregistration import require_preregistration_lock

    verification = require_preregistration_lock(
        require_exact_parent=not allow_descendant_head
    )
    if verification.get("lock_sha256") != EXPECTED_LOCK_SHA256:
        raise ValueError("full lock verifier returned an unexpected lock SHA-256")
    if int(verification.get("binding_count", -1)) != EXPECTED_BINDING_COUNT:
        raise ValueError("full lock verifier returned an unexpected binding count")

    run_summary = _load_json(RUN_SUMMARY_PATH)
    provenance = _load_json(PROVENANCE_PATH)
    if run_summary.get("preregistration_lock_sha256") != EXPECTED_LOCK_SHA256:
        raise ValueError("run summary does not authenticate the immutable lock")
    if provenance.get("preregistration_lock_sha256") != EXPECTED_LOCK_SHA256:
        raise ValueError("input provenance does not authenticate the immutable lock")
    if run_summary.get("status") != "COMPLETE" or provenance.get("status") != "COMPLETE":
        raise ValueError("formal core outputs are not COMPLETE")
    if run_summary.get("encoder_retrained") is not False:
        raise ValueError("encoder retraining firewall declaration is invalid")
    if run_summary.get("pcr_read") is not False:
        raise ValueError("pCR firewall declaration is invalid")

    return {
        "head": head,
        "lock_verification": verification,
        "lock": lock,
    }


def _import_locked_analyzer() -> ModuleType:
    module_name = "nonftv_locked_analyzer_for_presentation_erratum"
    spec = importlib.util.spec_from_file_location(module_name, ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load locked analyzer: {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(str(module.__file__)).resolve() != ANALYZER_PATH.resolve():
        raise ImportError("imported analyzer path differs from locked analyzer path")
    if _file_sha256(ANALYZER_PATH) != EXPECTED_ANALYZER_SHA256:
        raise ValueError("locked analyzer changed during import")
    return module


def _corrected_metric_sentence(row: Any) -> str:
    r2_label = str(row["natural_r2_metric"])
    r2_display = (
        "reconstructed target R²"
        if r2_label == "reconstructed_natural_r2"
        else "natural R²"
    )
    return (
        f"{row['arm']}/{row['representation']}，{row['endpoint']}："
        f"两 seed rho={row['seed_2026_spearman']:.3f}/{row['seed_3026_spearman']:.3f}，"
        f"{r2_display}={row['seed_2026_natural_r2']:.3f}/{row['seed_3026_natural_r2']:.3f}，"
        f"n={int(row['n_min'])}–{int(row['n_max'])}"
    )


def _normalize_delivery_fields(report_text: str) -> str:
    normalized: list[str] = []
    for line in report_text.splitlines(keepends=True):
        if line.startswith("- Reported experiment commit SHA："):
            ending = "\n" if line.endswith("\n") else ""
            normalized.append("- Reported experiment commit SHA：`<DELIVERY_COMMIT>`" + ending)
        elif line.startswith("- Push status："):
            ending = "\n" if line.endswith("\n") else ""
            normalized.append("- Push status：`<DELIVERY_STATUS>`" + ending)
        else:
            normalized.append(line)
    return "".join(normalized)


def _erratum_notice(wrapper_sha256: str) -> str:
    wrapper_path = _relative(Path(__file__))
    return (
        f"\n{NOTICE_HEADING}\n\n"
        "冻结分析器的 `_metric_sentence` 在计算 R² 标签后遗漏返回语句，导致 10 个"
        "描述性最佳摘要未显示。该问题在全部正式聚合结果与验证完成后才被发现。"
        f"交付工具 `{wrapper_path}`（SHA-256 `{wrapper_sha256}`）先完整验证原锁、"
        "133 个绑定与冻结分析器哈希，再仅在内存中恢复原本已经写在相邻不可达代码块中的"
        "返回格式，并调用冻结分析器的 `--report-only` 路径。\n\n"
        "这是锁定后、未纳入预注册的纯呈现勘误：没有重跑或重拟合任何模型，没有修改"
        "target、cohort、fold、representation、threshold、metric、gate、classification、"
        "scorecard、figure 或 recommendation；private OOF 只做压缩文件字节哈希校验，"
        "从未被本工具打开解析。完整前后哈希与交付链记录在 `errata/` 的只读 JSON manifest。\n"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_scientific_conclusions() -> dict[str, Any]:
    gates = _load_json(GATES_PATH)
    observed_gates = {
        gate: bool(gates["gates"][gate]["passed"])
        for gate in ("A", "B", "C", "D", "E")
    }
    if observed_gates != EXPECTED_GATE_RESULTS:
        raise ValueError(f"primary gate results drifted: {observed_gates}")

    score_rows = _read_csv_rows(SCORECARD_PATH)
    observed_classifications = {
        row["target"]: row["scientific_classification"] for row in score_rows
    }
    if observed_classifications != EXPECTED_CLASSIFICATIONS:
        raise ValueError("scientific classification results drifted")
    recommendation_rows = _read_csv_rows(RECOMMENDATION_PATH)
    if len(recommendation_rows) != 1:
        raise ValueError("recommendation table must contain exactly one row")
    observed_recommendation = recommendation_rows[0]["recommendation"]
    if observed_recommendation != EXPECTED_RECOMMENDATION:
        raise ValueError("final recommendation drifted")
    return {
        "primary_gates": observed_gates,
        "scientific_classifications": observed_classifications,
        "recommendation": observed_recommendation,
    }


def _verify_corrected_report(
    analyzer: ModuleType,
    report_text: str,
    *,
    commit_sha: str,
    push_status: str,
    push_error: str,
    wrapper_sha256: str,
) -> None:
    if report_text.count("None") != 0:
        raise ValueError("corrected report still contains an unrendered missing-value literal")
    question_count = len(re.findall(r"^### (?:[1-9]|1[0-5])\.", report_text, flags=re.MULTILINE))
    if question_count != 15:
        raise ValueError(f"corrected report contains {question_count}, not 15, numbered questions")
    if report_text.count(NOTICE_HEADING) != 1:
        raise ValueError("corrected report must contain exactly one erratum notice")
    if wrapper_sha256 not in report_text:
        raise ValueError("corrected report does not authenticate the erratum wrapper")

    best = analyzer._read_csv(BEST_CELLS_PATH)
    requested = (
        ("static_raw_early", "FTV"),
        ("static_raw_early", "LD"),
        ("static_residual_early", "LD"),
        ("static_raw_early", "SPH"),
        ("static_residual_early", "SPH"),
        ("static_raw_early", "BPE"),
        ("static_residual_early", "BPE"),
        ("dynamic_residual_early", "LD"),
        ("dynamic_residual_early", "SPH"),
        ("dynamic_residual_early", "BPE"),
    )
    for analysis, target in requested:
        row = analyzer._best_lookup(best, analysis, target)
        sentence = _corrected_metric_sentence(row)
        if report_text.count(sentence) != 1:
            raise ValueError(
                f"corrected report does not contain exactly one expected summary: {analysis}/{target}"
            )

    expected_commit_line = f"- Reported experiment commit SHA：`{commit_sha}`"
    if expected_commit_line not in report_text:
        raise ValueError("corrected report commit disclosure differs from the invocation")
    if push_status == "GITHUB_PUSH_FAILED":
        expected_push = f"- Push status：`GITHUB_PUSH_FAILED；真实错误：`{push_error}``"
        # The locked renderer nests Markdown backticks around a real error.
        if f"- Push status：`GITHUB_PUSH_FAILED" not in report_text or push_error not in report_text:
            raise ValueError("corrected report push-failure disclosure differs from the invocation")
        del expected_push
    elif f"- Push status：`{push_status}`" not in report_text:
        raise ValueError("corrected report push-status disclosure differs from the invocation")


def _load_erratum_manifests() -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(ERRATA_ROOT.glob("*.json")):
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise PermissionError(f"erratum manifest must be a regular read-only file: {path}")
        payload = _load_json(path)
        if payload.get("schema_version") not in {
            BASE_MANIFEST_SCHEMA,
            DELIVERY_MANIFEST_SCHEMA,
        }:
            raise ValueError(f"unknown erratum manifest schema: {path}")
        manifests.append((path, payload))
    return manifests


def _verify_base_manifest(
    *,
    wrapper_sha256: str,
    scientific_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if not BASE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("base presentation erratum manifest is missing")
    payload = _load_json(BASE_MANIFEST_PATH)
    if payload.get("schema_version") != BASE_MANIFEST_SCHEMA:
        raise ValueError("base presentation erratum manifest schema differs")
    if payload.get("post_lock_unbound_presentation_only") is not True:
        raise ValueError("base manifest does not disclose post-lock unbound status")
    if payload.get("preregistration_lock", {}).get("sha256") != EXPECTED_LOCK_SHA256:
        raise ValueError("base manifest lock identity differs")
    if payload.get("locked_analyzer", {}).get("sha256") != EXPECTED_ANALYZER_SHA256:
        raise ValueError("base manifest analyzer identity differs")
    if payload.get("erratum_wrapper", {}).get("sha256") != wrapper_sha256:
        raise ValueError("base manifest wrapper identity differs")
    observed_snapshot_sha = _canonical_sha256(scientific_snapshot)
    if payload.get("scientific_invariant_snapshot_sha256") != observed_snapshot_sha:
        raise ValueError("scientific artifact snapshot differs from the base erratum manifest")
    if payload.get("scientific_invariant_snapshot") != scientific_snapshot:
        raise ValueError("scientific artifact inventory differs from the base erratum manifest")
    return payload


def _manifest_target(commit_sha: str, push_status: str) -> Path:
    if commit_sha == "PENDING_LOCAL_COMMIT" and push_status == "NOT_ATTEMPTED":
        return BASE_MANIFEST_PATH
    return ERRATA_ROOT / f"report_delivery_{commit_sha[:12]}_{push_status.lower()}.json"


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if "\n" in arguments.push_error or "\r" in arguments.push_error:
        raise ValueError("--push-error must be a single line; preserve multiline output with literal \\n")
    if arguments.push_status == "GITHUB_PUSH_FAILED" and not arguments.push_error.strip():
        raise ValueError("GITHUB_PUSH_FAILED requires the real --push-error")
    if arguments.push_status != "GITHUB_PUSH_FAILED" and arguments.push_error:
        raise ValueError("--push-error is allowed only with GITHUB_PUSH_FAILED")
    is_pending = arguments.commit_sha == "PENDING_LOCAL_COMMIT"
    if is_pending:
        if arguments.push_status != "NOT_ATTEMPTED" or arguments.allow_descendant_head:
            raise ValueError("pending render requires NOT_ATTEMPTED at the exact parent HEAD")
    else:
        if re.fullmatch(r"[0-9a-f]{40}", arguments.commit_sha) is None:
            raise ValueError("--commit-sha must be PENDING_LOCAL_COMMIT or a full lowercase Git SHA")
        if arguments.push_status == "NOT_ATTEMPTED":
            raise ValueError("a committed delivery render must record PUSHED or GITHUB_PUSH_FAILED")
        if not arguments.allow_descendant_head:
            raise ValueError("committed delivery render requires --allow-descendant-head")
        status_code = subprocess.run(
            ["git", "merge-base", "--is-ancestor", arguments.commit_sha, "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if status_code != 0:
            raise ValueError("reported experiment commit is not an ancestor of current HEAD")


def _apply(arguments: argparse.Namespace) -> dict[str, Any]:
    _validate_arguments(arguments)
    wrapper_path = Path(__file__).resolve()
    wrapper_sha256 = _file_sha256(wrapper_path)
    static = _verify_static_contract(
        allow_descendant_head=bool(arguments.allow_descendant_head)
    )
    scientific_before = _scientific_snapshot()
    scientific_snapshot_sha256 = _canonical_sha256(scientific_before)
    conclusions_before = _verify_scientific_conclusions()
    report_before = REPORT_PATH.read_bytes()
    report_before_mode = stat.S_IMODE(REPORT_PATH.stat().st_mode)
    report_before_text = report_before.decode("utf-8")

    existing_manifests = _load_erratum_manifests()
    is_initial = not BASE_MANIFEST_PATH.exists()
    if is_initial:
        if existing_manifests:
            raise ValueError("delivery manifests exist without a base presentation erratum")
        if arguments.commit_sha != "PENDING_LOCAL_COMMIT":
            raise ValueError("the first erratum render must precede the scientific commit")
        if report_before_text.count("None") != BUGGY_SUMMARY_COUNT:
            raise ValueError(
                f"initial report has {report_before_text.count('None')}, not ten, missing summaries"
            )
        previous_manifest: dict[str, Any] | None = None
    else:
        base_manifest = _verify_base_manifest(
            wrapper_sha256=wrapper_sha256,
            scientific_snapshot=scientific_before,
        )
        if report_before_text.count("None") != 0:
            raise ValueError("an already corrected report regained missing summaries")
        normalized_before_sha256 = _sha256_bytes(
            _normalize_delivery_fields(report_before_text).encode("utf-8")
        )
        if normalized_before_sha256 != base_manifest["report"]["normalized_corrected_sha256"]:
            raise ValueError("current report differs beyond delivery-only fields")
        matching = [
            (path, payload)
            for path, payload in existing_manifests
            if payload.get("report", {}).get("after_sha256") == _sha256_bytes(report_before)
        ]
        if len(matching) != 1:
            raise ValueError("current report does not have exactly one manifest-chain predecessor")
        previous_path, previous_payload = matching[0]
        previous_manifest = {
            "path": _relative(previous_path),
            "sha256": _file_sha256(previous_path),
            "schema_version": previous_payload["schema_version"],
        }

    target_manifest = _manifest_target(arguments.commit_sha, arguments.push_status)
    if target_manifest.exists() or target_manifest.is_symlink():
        raise FileExistsError(f"target delivery manifest already exists: {target_manifest}")

    analyzer = _import_locked_analyzer()
    sample = analyzer.pd.Series(
        {
            "natural_r2_metric": "natural_r2",
            "arm": "LOCAL0",
            "representation": "Z1",
            "endpoint": "T0",
            "seed_2026_spearman": 0.1,
            "seed_3026_spearman": 0.2,
            "seed_2026_natural_r2": 0.01,
            "seed_3026_natural_r2": 0.02,
            "n_min": 1,
            "n_max": 1,
        }
    )
    if analyzer._metric_sentence(sample) is not None:
        raise ValueError("known frozen analyzer bug signature is absent")

    old_metric_sentence = analyzer._metric_sentence
    old_argv = list(sys.argv)
    analyzer._metric_sentence = _corrected_metric_sentence
    analyzer_arguments = [
        str(ANALYZER_PATH),
        "--report-only",
        "--commit-sha",
        arguments.commit_sha,
        "--push-status",
        arguments.push_status,
    ]
    if arguments.push_error:
        analyzer_arguments.extend(["--push-error", arguments.push_error])
    if arguments.allow_descendant_head:
        analyzer_arguments.append("--allow-descendant-head")

    try:
        sys.argv = analyzer_arguments
        analyzer.main()
        rendered = REPORT_PATH.read_text(encoding="utf-8")
        if NOTICE_HEADING in rendered:
            raise ValueError("locked report-only renderer unexpectedly emitted the erratum notice")
        corrected = rendered.rstrip() + "\n" + _erratum_notice(wrapper_sha256)
        _atomic_replace_bytes(REPORT_PATH, corrected.encode("utf-8"), 0o644)

        scientific_after = _scientific_snapshot()
        if scientific_after != scientific_before:
            before_paths = set(scientific_before)
            after_paths = set(scientific_after)
            changed = sorted(
                path
                for path in before_paths & after_paths
                if scientific_before[path] != scientific_after[path]
            )
            raise ValueError(
                "non-report artifacts changed during presentation erratum: "
                f"added={sorted(after_paths-before_paths)}, "
                f"removed={sorted(before_paths-after_paths)}, changed={changed}"
            )
        conclusions_after = _verify_scientific_conclusions()
        if conclusions_after != conclusions_before:
            raise ValueError("scientific conclusions changed during presentation erratum")
        if _file_sha256(LOCK_PATH) != EXPECTED_LOCK_SHA256:
            raise ValueError("immutable lock changed during presentation erratum")
        if _file_sha256(ANALYZER_PATH) != EXPECTED_ANALYZER_SHA256:
            raise ValueError("locked analyzer changed during presentation erratum")
        corrected_text = REPORT_PATH.read_text(encoding="utf-8")
        _verify_corrected_report(
            analyzer,
            corrected_text,
            commit_sha=arguments.commit_sha,
            push_status=arguments.push_status,
            push_error=arguments.push_error,
            wrapper_sha256=wrapper_sha256,
        )
        normalized_corrected_sha256 = _sha256_bytes(
            _normalize_delivery_fields(corrected_text).encode("utf-8")
        )
        if not is_initial:
            base_manifest = _load_json(BASE_MANIFEST_PATH)
            if normalized_corrected_sha256 != base_manifest["report"]["normalized_corrected_sha256"]:
                raise ValueError("delivery rerender changed scientific report content")

        manifest = {
            "schema_version": (
                BASE_MANIFEST_SCHEMA if is_initial else DELIVERY_MANIFEST_SCHEMA
            ),
            "status": "COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "post_lock_unbound_presentation_only": True,
            "not_preregistered_analysis_code": True,
            "invocation": analyzer_arguments[1:],
            "git": {
                "branch": EXPECTED_BRANCH,
                "head": static["head"],
                "locked_parent": EXPECTED_PARENT_SHA,
                "reported_experiment_commit_sha": arguments.commit_sha,
                "push_status": arguments.push_status,
                "push_error": arguments.push_error,
            },
            "preregistration_lock": {
                "path": _relative(LOCK_PATH),
                "sha256": EXPECTED_LOCK_SHA256,
                "mode": "0444",
                "binding_count": EXPECTED_BINDING_COUNT,
                "verification": static["lock_verification"]["status"],
            },
            "locked_analyzer": {
                "path": _relative(ANALYZER_PATH),
                "sha256": EXPECTED_ANALYZER_SHA256,
                "modified_on_disk": False,
                "known_bug": (
                    "_metric_sentence computes the R2 label but omits its return; "
                    "the intended return block is unreachable after _interval_sentence"
                ),
            },
            "erratum_wrapper": {
                "path": _relative(wrapper_path),
                "sha256": wrapper_sha256,
                "in_memory_patch_only": True,
            },
            "correction_scope": {
                "descriptive_summary_sentences_restored": BUGGY_SUMMARY_COUNT,
                "literal_missing_summary_count_before": (
                    report_before_text.count("None")
                ),
                "literal_missing_summary_count_after": corrected_text.count("None"),
                "public_artifact_manifest_bullet_reordered_by_locked_report_only_path": is_initial,
                "transparent_erratum_notice_appended": True,
                "model_rerun": False,
                "model_refit": False,
                "private_oof_parsed": False,
                "thresholds_changed": False,
                "metrics_changed": False,
                "gates_changed": False,
                "classifications_changed": False,
                "figures_changed": False,
                "recommendation_changed": False,
            },
            "scientific_conclusions": conclusions_after,
            "scientific_invariant_snapshot_sha256": scientific_snapshot_sha256,
            "scientific_invariant_snapshot": scientific_before,
            "private_oof": {
                "path": _relative(PRIVATE_OOF_PATH),
                "sha256": scientific_before[_relative(PRIVATE_OOF_PATH)]["sha256"],
                "mode": scientific_before[_relative(PRIVATE_OOF_PATH)]["mode"],
                "parsed": False,
            },
            "report": {
                "path": _relative(REPORT_PATH),
                "before_sha256": _sha256_bytes(report_before),
                "after_sha256": _file_sha256(REPORT_PATH),
                "before_mode": f"{report_before_mode:04o}",
                "after_mode": f"{stat.S_IMODE(REPORT_PATH.stat().st_mode):04o}",
                "normalized_corrected_sha256": normalized_corrected_sha256,
                "delivery_fields_excluded_from_normalization": [
                    "Reported experiment commit SHA",
                    "Push status",
                ],
            },
            "previous_manifest": previous_manifest,
        }
        _atomic_json_no_replace(target_manifest, manifest)
    except BaseException:
        _atomic_replace_bytes(REPORT_PATH, report_before, report_before_mode)
        raise
    finally:
        analyzer._metric_sentence = old_metric_sentence
        sys.argv = old_argv

    return {
        "status": "COMPLETE",
        "manifest": _relative(target_manifest),
        "manifest_sha256": _file_sha256(target_manifest),
        "report_sha256": _file_sha256(REPORT_PATH),
        "normalized_report_sha256": manifest["report"]["normalized_corrected_sha256"],
        "scientific_snapshot_sha256": scientific_snapshot_sha256,
        "primary_gates": conclusions_after["primary_gates"],
        "recommendation": conclusions_after["recommendation"],
        "private_oof_parsed": False,
    }


def _verify_existing(arguments: argparse.Namespace) -> dict[str, Any]:
    if not BASE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("cannot verify before the base erratum has been applied")
    wrapper_sha256 = _file_sha256(Path(__file__).resolve())
    static = _verify_static_contract(
        allow_descendant_head=bool(arguments.allow_descendant_head)
    )
    scientific = _scientific_snapshot()
    base = _verify_base_manifest(
        wrapper_sha256=wrapper_sha256,
        scientific_snapshot=scientific,
    )
    conclusions = _verify_scientific_conclusions()
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    normalized_sha256 = _sha256_bytes(
        _normalize_delivery_fields(report_text).encode("utf-8")
    )
    if normalized_sha256 != base["report"]["normalized_corrected_sha256"]:
        raise ValueError("current report differs beyond delivery-only fields")
    manifests = _load_erratum_manifests()
    current_report_sha256 = _file_sha256(REPORT_PATH)
    matching = [
        _relative(path)
        for path, payload in manifests
        if payload.get("report", {}).get("after_sha256") == current_report_sha256
    ]
    if len(matching) != 1:
        raise ValueError("current report does not have exactly one immutable render manifest")
    if report_text.count("None") != 0 or report_text.count(NOTICE_HEADING) != 1:
        raise ValueError("current report presentation erratum is incomplete")
    if _file_sha256(ANALYZER_PATH) != EXPECTED_ANALYZER_SHA256:
        raise ValueError("locked analyzer changed during erratum verification")
    return {
        "status": "PASS",
        "head": static["head"],
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "analyzer_sha256": EXPECTED_ANALYZER_SHA256,
        "wrapper_sha256": wrapper_sha256,
        "report_sha256": current_report_sha256,
        "normalized_report_sha256": normalized_sha256,
        "current_render_manifest": matching[0],
        "scientific_snapshot_sha256": _canonical_sha256(scientific),
        "primary_gates": conclusions["primary_gates"],
        "recommendation": conclusions["recommendation"],
        "private_oof_parsed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", default="PENDING_LOCAL_COMMIT")
    parser.add_argument(
        "--push-status",
        choices=("NOT_ATTEMPTED", "PUSHED", "GITHUB_PUSH_FAILED"),
        default="NOT_ATTEMPTED",
    )
    parser.add_argument("--push-error", default="")
    parser.add_argument("--allow-descendant-head", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the current corrected report and immutable manifest chain",
    )
    arguments = parser.parse_args()
    if arguments.verify_only:
        result = _verify_existing(arguments)
    else:
        result = _apply(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
