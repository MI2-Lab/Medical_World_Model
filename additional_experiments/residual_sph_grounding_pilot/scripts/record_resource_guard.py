#!/usr/bin/env python3
"""Record why the frozen formal matrix was not launched on occupied GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


def _snapshot() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        index, total, used, free = (int(part.strip()) for part in line.split(","))
        rows.append(
            {"device": index, "total_mib": total, "used_mib": used, "free_mib": free}
        )
    if not rows:
        raise RuntimeError("no CUDA devices were reported")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=60_000)
    args = parser.parse_args()
    if args.minimum_free_mib <= 0:
        raise ValueError("minimum-free-mib must be positive")
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    completed = list(
        (EXPERIMENT_ROOT / "checkpoints" / "formal_4x8").glob(
            "seed_*/S*/fold_*/selected.pt"
        )
    )
    if completed:
        raise RuntimeError("resource-guard status is only valid before all formal cells")
    snapshot = _snapshot()
    if any(row["free_mib"] >= args.minimum_free_mib for row in snapshot):
        raise RuntimeError("a GPU meets the threshold; use the frozen matrix runner")
    observed = datetime.now(timezone.utc).isoformat()
    execution = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD",
        "observed_at_utc": observed,
        "reason": (
            "Every requested CUDA device was below the preregistered free-memory "
            "threshold; launching LOCAL 3-D training would risk contention."
        ),
        "gpu_snapshot": snapshot,
        "minimum_free_mib_required_by_runner": int(args.minimum_free_mib),
        "pre_existing_processes_terminated": False,
        "formal_new_training_cells_required": 30,
        "formal_new_training_cells_completed": 0,
        "scientific_gates_evaluated": False,
        "classification_assigned": False,
    }
    decision = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD",
        "decision_is_scientific_result": False,
        "gates": {
            "A_RESPONSE_SAFETY": "NOT_EVALUATED",
            "B_RESIDUAL_SPH_ORGANIZATION": "NOT_EVALUATED",
            "C_RESIDUAL_BENEFIT_OVER_RAW": "NOT_EVALUATED",
            "D_DOWNSTREAM_COMPLEMENTARITY": "NOT_EVALUATED",
        },
        "classification": "NOT_ASSIGNED",
        "five_seed_confirmation_justified": "NOT_EVALUATED",
        "reason": (
            "The frozen implementation is ready, but 0/30 new training cells ran "
            "because no CUDA device met the safe free-memory threshold."
        ),
        "next_authorized_step": (
            "Run the frozen matrix unchanged when one or more requested GPUs meet "
            "the threshold; do not read pCR until representation_freeze.json exists."
        ),
    }
    for path, payload in (
        (EXPERIMENT_ROOT / "metrics/execution_status.json", execution),
        (EXPERIMENT_ROOT / "metrics/decision.json", decision),
    ):
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("status") != "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD":
                raise FileExistsError(f"refusing to overwrite completed artifact: {path}")
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": execution["status"], "gpu_snapshot": snapshot}))


if __name__ == "__main__":
    main()
