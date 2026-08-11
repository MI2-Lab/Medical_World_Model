from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT_ROOT / "configs" / "experiment.json"
MODULE_PATH = EXPERIMENT_ROOT / "scripts" / "inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("classical_dce_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _locked_sources_available() -> bool:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    keys = ("radiomics_workbook", "clinical_workbook", "mri_fold_manifest")
    return all(Path(payload["source"][key]).is_file() for key in keys)


pytestmark = pytest.mark.skipif(
    not _locked_sources_available(), reason="locked private source assets are unavailable"
)


def test_inventory_outputs_are_aggregate_only(tmp_path: Path) -> None:
    module = _load_module()
    result = module.run_inventory(CONFIG, tmp_path)

    inventory = pd.read_csv(result.inventory_path)
    missingness = pd.read_csv(result.missingness_path)
    report = result.report_path.read_text(encoding="utf-8")

    assert inventory.shape[0] == 29
    assert inventory["column"].is_unique
    assert inventory["n_missing"].eq(0).all()
    assert inventory["workbook_patient_coverage_n"].eq(384).all()
    assert inventory["mri_overlap_patient_coverage_n"].eq(375).all()
    assert inventory.groupby(["family", "role"]).size().to_dict() == {
        ("BPE", "absolute_measurement"): 4,
        ("BPE", "derived_baseline_percent_change"): 3,
        ("FTV", "absolute_measurement"): 4,
        ("FTV", "derived_baseline_percent_change"): 3,
        ("ID", "patient_identifier"): 1,
        ("LD", "absolute_measurement"): 4,
        ("LD", "derived_baseline_percent_change"): 3,
        ("SPH", "absolute_measurement"): 4,
        ("SPH", "derived_baseline_percent_change"): 3,
    }
    assert missingness.shape[0] == 87
    assert set(missingness["scope"]) == {
        "source_workbook",
        "complete4_mri_cohort",
        "mri_matched_reference",
    }
    mri_scope = missingness.loc[
        missingness["scope"].eq("complete4_mri_cohort")
    ]
    assert mri_scope["n_patients"].eq(808).all()
    assert mri_scope["n_valid_patients"].eq(375).all()
    assert mri_scope["n_missing_patients"].eq(433).all()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_ids = pd.read_excel(
        config["source"]["radiomics_workbook"],
        sheet_name=config["source"]["radiomics_sheet"],
        usecols=["CLINICAL-TRIAL-SUBJECT-ID"],
    )["CLINICAL-TRIAL-SUBJECT-ID"].astype(str)
    output_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (result.inventory_path, result.missingness_path, result.report_path)
    )
    standalone_six_digit_tokens = set(
        re.findall(r"(?<![A-Za-z0-9])\d{6}(?![A-Za-z0-9])", output_text)
    )
    assert standalone_six_digit_tokens.isdisjoint(set(source_ids))
    assert "29.43%" in report
    assert "late/pre-surgery" in report


def test_sha_mismatch_fails_before_writing_outputs(tmp_path: Path) -> None:
    module = _load_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["source"]["radiomics_sha256"] = "0" * 64
    bad_config = tmp_path / "bad_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")
    output_root = tmp_path / "outputs"

    with pytest.raises(module.InventoryError, match="SHA-256 mismatch"):
        module.run_inventory(bad_config, output_root)

    assert not output_root.exists()
