#!/usr/bin/env python3
"""Run the outcome-blind transfer gate and create MECHANISM_LOCK on PASS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json  # noqa: E402
from dinov3_rg.locking import freeze_mechanism_lock, verify_representation_lock  # noqa: E402
from dinov3_rg.mechanism import mechanism_gate, mechanism_metrics  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel, public_artifact_privacy_scan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "features/private/states")
    parser.add_argument("--target-root", type=Path, default=ROOT / "features/private/fold_targets")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/formal")
    parser.add_argument("--overwrite-lock", action="store_true")
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    representation = verify_representation_lock()
    metrics = mechanism_metrics(args.state_root, args.target_root)
    metrics_path = ROOT / "metrics/mechanism_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    gate = mechanism_gate(metrics, args.checkpoint_root)
    gate["representation_lock_content_sha256"] = representation["lock_content_sha256"]
    gate_path = ROOT / "mechanism_gate.json"
    atomic_json(gate_path, gate)
    privacy = public_artifact_privacy_scan(ROOT)
    if privacy["status"] != "PASS":
        raise RuntimeError(f"public artifact privacy gate failed: {privacy['failures']}")
    if gate["status"] != "PASS":
        optimization_keys = {
            "static_ftv_not_degraded",
            "delta_ftv_not_degraded",
            "state_noncollapse",
            "formal_d3_radiomics_gradients",
        }
        optimization_safe = all(gate["gates"][name] for name in optimization_keys)
        decision_class = (
            "RADIOMICS_NOT_TRANSFERRED" if optimization_safe else "GROUNDING_OPTIMIZATION_CONFLICT"
        )
        atomic_json(
            ROOT / "decision.json",
            {
                "schema_version": 1,
                "status": "TERMINATED_AT_MECHANISM_GATE",
                "decision_class": decision_class,
                "mechanism_gates": gate["gates"],
                "pcr_evaluation_started": False,
                "pcr_outcomes_read": False,
            },
        )
        atomic_json(
            ROOT / "acceptance_check.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "failed_stage": "mechanism_gate",
                "representation_cells_completed": 75,
                "mechanism_lock_created": False,
                "pcr_outcomes_read": False,
                "privacy_gate": "PASS",
                "decision_class": decision_class,
            },
        )
        raise SystemExit("mechanism gate NO-GO; pCR remains locked")
    lock = freeze_mechanism_lock(gate_path, metrics_path, overwrite=args.overwrite_lock)
    atomic_json(
        ROOT / "acceptance_check.json",
        {
            "schema_version": 1,
            "status": "MECHANISM_PASS_PCR_LOCKED",
            "representation_cells_completed": 75,
            "mechanism_lock_created": True,
            "pcr_outcomes_read": False,
            "privacy_gate": "PASS",
        },
    )
    print({"gate": gate, "lock": lock})


if __name__ == "__main__":
    main()
