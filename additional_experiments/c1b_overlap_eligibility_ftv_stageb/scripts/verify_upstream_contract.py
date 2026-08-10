#!/usr/bin/env python3
"""Verify immutable prior decisions and every reused C1B-H contract."""

from __future__ import annotations

import os
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    verify_preregistration,
    verify_upstream_contract,
)


def main() -> None:
    preregistration = verify_preregistration()
    observed = verify_upstream_contract()
    public = {
        "schema_version": 1,
        "status": "PASS",
        "preregistration_plan_sha256": preregistration["plan_sha256"],
        "prior_stage_a_no_go_immutable": True,
        "prior_audit_not_repairable_immutable": True,
        **observed,
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "metrics/upstream_contract_verification.json",
        json_text(public),
        overwrite=True,
    )
    print(json_text(public), end="")


if __name__ == "__main__":
    main()

