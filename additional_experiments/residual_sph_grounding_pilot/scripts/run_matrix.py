#!/usr/bin/env python3
"""Plan or execute the preregistered S1/S2/S2-L10 matrix safely."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


TRAIN = EXPERIMENT_ROOT / "scripts" / "train_cell.py"
EXPORT = EXPERIMENT_ROOT / "scripts" / "export_features.py"
CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints" / "formal_4x8"
FEATURES = EXPERIMENT_ROOT / "features" / "formal_4x8"
LOGS = EXPERIMENT_ROOT / "logs" / "formal_4x8"
PYTHON = Path(sys.executable).resolve()


def _cells(arms: tuple[str, ...]) -> list[tuple[str, int, int]]:
    return [(arm, seed, fold) for arm in arms for seed in (2026, 3026) for fold in range(5)]


def _command(
    mode: str,
    arm: str,
    seed: int,
    fold: int,
    lock_sha256: str,
) -> tuple[list[str], Path, Path]:
    run = CHECKPOINTS / f"seed_{seed}" / arm / f"fold_{fold}"
    feature = FEATURES / f"seed_{seed}" / arm / f"fold_{fold}" / "response_state.private.npz"
    if mode == "train":
        command = [
            str(PYTHON), str(TRAIN), "--arm", arm, "--seed-base", str(seed),
            "--fold", str(fold), "--output-dir", str(run), "--device", "cuda",
            "--preregistration-lock-sha256", lock_sha256,
        ]
        complete = run / "selected.pt"
    else:
        command = [
            str(PYTHON), str(EXPORT), "--arm", arm, "--seed-base", str(seed),
            "--fold", str(fold), "--checkpoint", str(run / "selected.pt"),
            "--output", str(feature), "--device", "cuda",
            "--preregistration-lock-sha256", lock_sha256,
        ]
        complete = feature
    log = LOGS / f"{mode}_seed_{seed}_{arm}_fold_{fold}.log"
    return command, complete, log


def _gpu_free_mib() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free = (part.strip() for part in line.split(",", maxsplit=1))
        output[int(index)] = int(free)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "train", "export"), default="plan")
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--arms", default="S1,S2,S2_L10")
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=60_000)
    args = parser.parse_args()
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    arms = tuple(value.strip().upper() for value in args.arms.split(",") if value.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a nonempty unique list")
    if not arms or not set(arms).issubset({"S1", "S2", "S2_L10"}):
        raise ValueError("arms are outside the preregistered matrix")
    if len(args.preregistration_lock_sha256) != 64:
        raise ValueError("preregistration lock must be a SHA-256 digest")
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    plan = []
    for arm, seed, fold in _cells(arms):
        command, complete, log = _command(
            "train" if args.mode == "plan" else args.mode,
            arm, seed, fold, args.preregistration_lock_sha256,
        )
        plan.append(
            {
                "arm": arm,
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "complete": complete.is_file(),
                "command": command,
                "log": str(log.relative_to(EXPERIMENT_ROOT)),
            }
        )
    if args.mode == "plan":
        public_plan = [
            {
                "arm": item["arm"],
                "seed_base": item["seed_base"],
                "fold": item["fold"],
                "effective_seed": item["effective_seed"],
                "complete": item["complete"],
                "log": item["log"],
            }
            for item in plan
        ]
        print(json.dumps({"cells": public_plan}, indent=2, sort_keys=True))
        return
    if not PYTHON.is_file():
        raise FileNotFoundError(f"required CUDA Python is missing: {PYTHON}")
    free = _gpu_free_mib()
    eligible_devices = tuple(
        device for device in devices if free.get(device, -1) >= args.minimum_free_mib
    )
    if not eligible_devices:
        unavailable = {device: free.get(device, -1) for device in devices}
        raise RuntimeError(
            "refusing to contend with active GPU jobs; insufficient free MiB: "
            + json.dumps(unavailable, sort_keys=True)
        )
    devices = eligible_devices
    LOGS.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = deque(item for item in plan if not item["complete"])
    running: dict[int, tuple[subprocess.Popen[bytes], object, dict[str, object]]] = {}
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            item = pending.popleft()
            log_path = EXPERIMENT_ROOT / str(item["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            stream = log_path.open("xb")
            os.chmod(log_path, 0o600)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            process = subprocess.Popen(
                item["command"],
                cwd=REPO_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            running[device] = process, stream, item
        if not running:
            break
        time.sleep(5)
        for device, (process, stream, item) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del running[device]
            if code:
                for other, (other_process, other_stream, _) in running.items():
                    other_process.terminate()
                    other_stream.close()
                raise RuntimeError(
                    f"matrix cell failed on GPU {device}: "
                    f"{item['arm']}/seed_{item['seed_base']}/fold_{item['fold']}"
                )
    print(json.dumps({"status": "COMPLETE", "mode": args.mode, "cells": len(plan)}, sort_keys=True))


if __name__ == "__main__":
    main()
