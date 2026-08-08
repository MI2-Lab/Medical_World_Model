#!/usr/bin/env python3
"""在多 GPU 上串行每卡、并行跨卡运行 selected 或 representative-last audit。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.assets import checkpoint_path  # noqa: E402
from gjca.contracts import FOLDS, SEED_BASES, atomic_csv, file_sha256  # noqa: E402
from gjca.gradients import output_path, validate_gradient_file  # noqa: E402
from gjca.source_contract import assert_source_contract  # noqa: E402


def _tasks(mode: str) -> list[tuple[int, int, str]]:
    if mode == "selected":
        return [(seed, fold, "selected") for seed in SEED_BASES for fold in FOLDS]
    representatives = pd.read_csv(ROOT / "metrics" / "representative_runs.csv")
    if (
        len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
    ):
        raise ValueError("representative run contract 错误")
    return [
        (int(row.seed_base), int(row.fold), "last")
        for row in representatives.itertuples(index=False)
    ]


def _run_gpu(
    gpu: int, tasks: Iterable[tuple[int, int, str]], resume: bool
) -> list[str]:
    outputs: list[str] = []
    for seed, fold, kind in tasks:
        destination = output_path(seed, fold, kind)
        checkpoint_kind = "best" if kind == "selected" else "last"
        checkpoint_sha = file_sha256(checkpoint_path(seed, fold, "g3", checkpoint_kind))
        if resume and destination.is_file():
            validate_gradient_file(destination, seed, fold, kind, checkpoint_sha)
            outputs.append(str(destination))
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts" / "extract_gradients.py"),
            "--seed-base",
            str(seed),
            "--fold",
            str(fold),
            "--checkpoint-kind",
            kind,
            "--device",
            f"cuda:{gpu}",
        ]
        environment = dict(os.environ)
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log = ROOT / "logs" / f"gradient_seed_{seed}_fold_{fold}_{kind}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"gradient job 失败: {seed}/{fold}/{kind}; 见 {log}")
        validate_gradient_file(destination, seed, fold, kind, checkpoint_sha)
        outputs.append(str(destination))
    return outputs


def _combine(tasks: list[tuple[int, int, str]], mode: str, overwrite: bool) -> Path:
    frames = [
        validate_gradient_file(output_path(seed, fold, kind), seed, fold, kind)
        for seed, fold, kind in tasks
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["seed_base", "fold", "checkpoint_kind", "split", "batch_index", "group"]
    )
    expected = len(tasks) * 2 * 8 * 7
    if (
        len(combined) != expected
        or combined.duplicated(
            ["seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"]
        ).any()
    ):
        raise ValueError("combined gradient matrix coverage 错误")
    destination = (
        ROOT
        / "metrics"
        / (
            "batch_gradient_metrics.csv"
            if mode == "selected"
            else "trajectory_batch_gradient_metrics.csv"
        )
    )
    rows = combined.to_dict(orient="records")
    atomic_csv(destination, rows, overwrite=overwrite)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("selected", "trajectory"), default="selected"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert_source_contract(full_content_hash=True, full_checkpoint_hash=True)
    tasks = _tasks(args.mode)
    assignments = {
        gpu: tasks[index :: len(args.gpus)] for index, gpu in enumerate(args.gpus)
    }
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(_run_gpu, gpu, assigned, args.resume): gpu
            for gpu, assigned in assignments.items()
            if assigned
        }
        for future in as_completed(futures):
            results.extend(future.result())
    destination = _combine(tasks, args.mode, False)
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": args.mode,
                "jobs": len(results),
                "combined": str(destination.relative_to(ROOT)),
                "combined_sha256": file_sha256(destination),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
