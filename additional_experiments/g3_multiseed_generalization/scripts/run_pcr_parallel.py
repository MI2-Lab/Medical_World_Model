#!/usr/bin/env python3
"""并行补全互不重叠的正式 pCR readout cells，并逐份执行完整验证。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgrs.features import (  # noqa: E402
    DEFAULT_FOLD_MANIFEST,
    EXPECTED_FOLD_MANIFEST_SHA256,
    MODELS,
    SEED_BASES,
    file_sha256,
)
from dgrs.pcr import validate_pcr_asset  # noqa: E402


@dataclass(frozen=True)
class Cell:
    seed_base: int
    fold: int
    model: str


CELLS = tuple(
    Cell(seed, fold, model)
    for seed in SEED_BASES
    for fold in range(5)
    for model in MODELS
)
SCRIPT = ROOT / "scripts" / "run_pcr_readouts.py"


def roots(cell: Cell) -> tuple[Path, Path, Path]:
    return (
        ROOT / "features" / f"seed_{cell.seed_base}",
        ROOT / "predictions" / "pcr_readouts" / f"seed_{cell.seed_base}",
        ROOT / "metrics" / "pcr_readouts" / f"seed_{cell.seed_base}",
    )


def assets(cell: Cell) -> tuple[Path, Path, Path]:
    _, prediction_root, metric_root = roots(cell)
    model_fold = Path(cell.model) / f"fold_{cell.fold}"
    return (
        prediction_root / model_fold / "test_predictions.csv",
        metric_root / model_fold / "selection_records.csv",
        metric_root / model_fold / "summary.json",
    )


def validate(cell: Cell, fold_manifest: Path) -> None:
    feature_root, prediction_root, metric_root = roots(cell)
    validate_pcr_asset(
        model_name=cell.model,
        fold=cell.fold,
        seed_base=cell.seed_base,
        feature_root=feature_root,
        prediction_root=prediction_root,
        metric_root=metric_root,
        fold_manifest=fold_manifest,
    )


def run_cell(cell: Cell, fold_manifest: Path) -> Cell:
    feature_root, prediction_root, metric_root = roots(cell)
    command = [
        sys.executable,
        str(SCRIPT),
        "--model",
        cell.model,
        "--fold",
        str(cell.fold),
        "--seed-base",
        str(cell.seed_base),
        "--feature-root",
        str(feature_root),
        "--prediction-root",
        str(prediction_root),
        "--metric-root",
        str(metric_root),
        "--fold-manifest",
        str(fold_manifest),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pCR cell 失败 seed={cell.seed_base}/fold={cell.fold}/{cell.model}: "
            f"{completed.stderr[-4000:]}"
        )
    validate(cell, fold_manifest)
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers 必须位于 1–12")
    fold_manifest = args.fold_manifest.expanduser().resolve(strict=True)
    if file_sha256(fold_manifest) != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError("fold manifest SHA 漂移")

    pending: list[Cell] = []
    verified = 0
    for cell in CELLS:
        paths = assets(cell)
        exists = tuple(path.exists() for path in paths)
        if all(exists):
            validate(cell, fold_manifest)
            verified += 1
        elif any(exists):
            raise FileExistsError(f"发现 partial pCR asset，拒绝并行续跑：{paths}")
        else:
            pending.append(cell)
    print(
        json.dumps(
            {
                "status": "preflight_pass",
                "verified_complete": verified,
                "pending": len(pending),
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    completed_count = verified
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="pcr-cell") as pool:
        futures = {pool.submit(run_cell, cell, fold_manifest): cell for cell in pending}
        for future in as_completed(futures):
            cell = future.result()
            completed_count += 1
            print(
                json.dumps(
                    {
                        "status": "created_and_verified",
                        "seed_base": cell.seed_base,
                        "fold": cell.fold,
                        "model": cell.model,
                        "completed": completed_count,
                        "total": len(CELLS),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if completed_count != len(CELLS):
        raise AssertionError("pCR cell coverage 不完整")
    print(
        json.dumps(
            {"status": "pcr_parallel_complete", "completed": completed_count},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
