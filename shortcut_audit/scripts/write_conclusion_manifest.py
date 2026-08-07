#!/usr/bin/env python3
"""为 GitHub 轻量结论包生成可携带的 SHA256 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c413ec86af04795434bdc19e65bbb006c966f379"
CONCLUSION_FILES = (
    "README.md",
    "configs/audit_protocol_v1.yaml",
    "configs/retrain_paper_v1.yaml",
    "report/repository_inspection.md",
    "report/cache_compatibility.md",
    "report/reproduction_asset_contract.md",
    "report/retraining_protocol.md",
    "report/shortcut_audit_report.md",
    "metrics/prerequisite_check.json",
    "metrics/fold_manifest_validation.json",
    "metrics/response_cache_validation.json",
    "metrics/fivefold_training_validation.json",
    "metrics/prediction_coverage.csv",
    "metrics/fold_metrics.csv",
    "metrics/fold_summary.csv",
    "metrics/pooled_oof.csv",
    "metrics/native_differences.csv",
    "metrics/fold_changes.csv",
    "metrics/paired_bootstrap.csv",
    "metrics/probability_change_summary.csv",
    "metrics/copy_fold_metrics.csv",
    "metrics/copy_fold_summary.csv",
    "metrics/copy_pooled_metrics.csv",
    "metrics/copy_bootstrap.csv",
    "metrics/perturbation_latent_fold_metrics.csv",
    "metrics/perturbation_latent_fold_summary.csv",
    "metrics/perturbation_latent_pooled.csv",
    "metrics/donor_matching_summary.csv",
    "metrics/donor_matching_summary.json",
    "metrics/repetition_metrics.csv",
    "metrics/repetition_summary.csv",
    "figures/required_figures_manifest.json",
    "figures/01_native_perturbation_auroc.png",
    "figures/02_fold_auroc_change.png",
    "figures/03_learned_vs_copy_error.png",
    "figures/04_transition_gain_distribution.png",
    "figures/05_repeated_t0_probability_change.png",
    "figures/06_temporal_swap_probability_change.png",
    "figures/07_followup_swap_probability_change.png",
    "figures/08_f1_f5_native_comparison.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--allow-manifest", action="store_true")
    args = parser.parse_args()
    if not args.allow_manifest:
        raise SystemExit("结论包 manifest 未生成：必须显式添加 --allow-manifest")

    root = args.audit_root.resolve()
    output = root / "metrics" / "conclusion_artifacts_manifest.json"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有结论包 manifest：{output}")
    missing = [relative for relative in CONCLUSION_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"结论包文件不完整：{missing}")

    artifacts: list[dict[str, object]] = []
    for relative in CONCLUSION_FILES:
        path = root / relative
        artifact: dict[str, object] = {
            "path": relative,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            artifact["rows"] = len(pd.read_csv(path))
        artifacts.append(artifact)

    payload = {
        "schema_version": "shortcut_audit.github_conclusion_bundle.v1",
        "github_branch": "shortcut-audit",
        "source_repository_commit": SOURCE_COMMIT,
        "result_identity": "fivefold audit retraining; not original checkpoint reproduction",
        "scope": (
            "de-identified aggregate metrics, figures, configs and Chinese reports; "
            "audit source code and tests are versioned separately in the same branch"
        ),
        "excluded_local_artifact_classes": [
            "patient-level predictions and probability changes",
            "bootstrap draws",
            "donor-recipient mappings and latent diagnostics",
            "checkpoints, serialized readouts and tensor caches",
            "training and evaluation logs",
        ],
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
