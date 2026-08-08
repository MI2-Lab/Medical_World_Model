#!/usr/bin/env python3
"""Fail-closed validation for the frozen Stage-A input-contract config."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "stage_a.json"
VISITS = ["T0", "T1", "T2", "T3"]


EXPECTED_CONTRACTS: dict[str, dict[str, Any]] = {
    "C0": {
        "anchor_policy": "legacy_normalized_index_T0_projection",
        "reference_visits": ["T0"],
        "causal_deployability": "LEGACY_REFERENCE",
        "audit_only": False,
        "model_eligible": False,
        "uses_future_support": False,
        "views": ["legacy"],
    },
    "C1A": {
        "anchor_policy": "current_visit_support_center",
        "reference_visits": ["CURRENT"],
        "causal_deployability": "CURRENT_VISIT_CAUSAL_TEMPORAL_RECENTER_RISK",
        "audit_only": False,
        "model_eligible": True,
        "uses_future_support": False,
        "views": ["detail"],
    },
    "C1B": {
        "anchor_policy": "T0_support_center_and_frame",
        "reference_visits": ["T0"],
        "causal_deployability": "T0_ANCHORED_CAUSAL",
        "audit_only": False,
        "model_eligible": True,
        "uses_future_support": False,
        "views": ["detail"],
    },
    "C1C": {
        "anchor_policy": "T0_T3_union_bbox",
        "reference_visits": VISITS,
        "causal_deployability": "AUDIT_ONLY_FUTURE_INFORMATION",
        "audit_only": True,
        "model_eligible": False,
        "uses_future_support": True,
        "views": ["detail"],
    },
    "C2A": {
        "anchor_policy": "current_visit_support_center",
        "reference_visits": ["CURRENT"],
        "causal_deployability": "CURRENT_VISIT_CAUSAL_TEMPORAL_RECENTER_RISK",
        "audit_only": False,
        "model_eligible": True,
        "uses_future_support": False,
        "views": ["detail", "context"],
    },
    "C2B": {
        "anchor_policy": "T0_support_center_and_frame",
        "reference_visits": ["T0"],
        "causal_deployability": "T0_ANCHORED_CAUSAL",
        "audit_only": False,
        "model_eligible": True,
        "uses_future_support": False,
        "views": ["detail", "context"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage-A JSON config; defaults to configs/stage_a.json.",
    )
    return parser.parse_args()


def _mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} 必须是 object")
        return {}
    return value


def _bool(value: Any, path: str, errors: list[str]) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{path} 必须是 boolean")
        return None
    return value


def _positive_triplet(value: Any, path: str, errors: list[str]) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"{path} 必须是长度为 3 的数组")
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        errors.append(f"{path} 必须只含数值")
        return None
    output = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0.0 for item in output):
        errors.append(f"{path} 必须只含有限正数")
        return None
    return output


def _positive_shape(value: Any, path: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"{path} 必须是长度为 3 的数组")
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        errors.append(f"{path} 必须只含正整数")
        return None
    return [int(item) for item in value]


def _same_value(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    # Exact comparison is intentional for the enum/list/bool contract.  It also
    # prevents truthy integers from masquerading as JSON booleans.
    if type(actual) is not type(expected) or actual != expected:  # noqa: E721
        errors.append(f"{path} 必须为 {expected!r}，实际为 {actual!r}")


def _validate_grid(
    physical: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> dict[str, list[float] | list[int]] | None:
    if prefix == "detail":
        shape_key = "detail_output_shape_zyx"
        spacing_key = "detail_common_spacing_xyz_mm"
        fov_key = "detail_nominal_fov_xyz_mm"
    elif prefix == "context":
        shape_key = "context_output_shape_zyx"
        spacing_key = "context_effective_spacing_xyz_mm"
        fov_key = "context_nominal_fov_xyz_mm"
    else:  # pragma: no cover - internal programming guard
        raise ValueError(prefix)

    shape = _positive_shape(physical.get(shape_key), f"physical_crop.{shape_key}", errors)
    spacing = _positive_triplet(
        physical.get(spacing_key), f"physical_crop.{spacing_key}", errors
    )
    fov = _positive_triplet(physical.get(fov_key), f"physical_crop.{fov_key}", errors)
    if shape is None or spacing is None or fov is None:
        return None

    shape_xyz = list(reversed(shape))
    derived_fov = [float(length * step) for length, step in zip(shape_xyz, spacing, strict=True)]
    for axis, observed, expected in zip("XYZ", fov, derived_fov, strict=True):
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-8):
            errors.append(
                f"{prefix} {axis} 轴 FOV 不一致：配置 {observed} mm，"
                f"shape×spacing={expected} mm"
            )
    return {
        "shape_zyx": shape,
        "shape_xyz": shape_xyz,
        "spacing_xyz_mm": spacing,
        "fov_xyz_mm": fov,
        "derived_fov_xyz_mm": derived_fov,
    }


def validate_config(payload: Any) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    config = _mapping(payload, "root", errors)
    if not config:
        return errors, {}

    if type(config.get("schema_version")) is not int or config.get("schema_version") != 1:
        errors.append("schema_version 必须严格为整数 1")

    cohort = _mapping(config.get("cohort"), "cohort", errors)
    _same_value(cohort.get("visits"), VISITS, "cohort.visits", errors)
    _same_value(cohort.get("outcome_free"), True, "cohort.outcome_free", errors)

    geometry = _mapping(config.get("geometry"), "geometry", errors)
    _same_value(geometry.get("index_order"), "XYZ", "geometry.index_order", errors)
    _same_value(geometry.get("tensor_order"), "CZYX", "geometry.tensor_order", errors)
    _same_value(
        geometry.get("voxel_footprint_containment"),
        True,
        "geometry.voxel_footprint_containment",
        errors,
    )
    registration_sensitivity = _bool(
        geometry.get("image_only_rigid_registration_sensitivity_completed"),
        "geometry.image_only_rigid_registration_sensitivity_completed",
        errors,
    )
    orientation_contract_validated = _bool(
        geometry.get("production_orientation_canonicalization_validated"),
        "geometry.production_orientation_canonicalization_validated",
        errors,
    )

    physical = _mapping(config.get("physical_crop"), "physical_crop", errors)
    detail = _validate_grid(physical, "detail", errors)
    context = _validate_grid(physical, "context", errors)

    margins = physical.get("margin_candidates_mm")
    if not isinstance(margins, list) or not margins or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (margins if isinstance(margins, list) else [])
    ):
        errors.append("physical_crop.margin_candidates_mm 必须是有限非负数数组")
    elif [float(value) for value in margins] != sorted(set(float(value) for value in margins)):
        errors.append("physical_crop.margin_candidates_mm 必须严格递增且无重复")
    selected_margin = physical.get("selected_margin_mm")
    if (
        isinstance(selected_margin, bool)
        or not isinstance(selected_margin, (int, float))
        or not isinstance(margins, list)
        or float(selected_margin) not in [float(value) for value in margins]
    ):
        errors.append("physical_crop.selected_margin_mm 必须来自 margin_candidates_mm")

    _same_value(
        physical.get("expand_instead_of_truncate"),
        True,
        "physical_crop.expand_instead_of_truncate",
        errors,
    )
    _same_value(
        physical.get("expanded_case_policy"),
        "construct_full_physical_support_then_fixed_shape_resample_and_flag_scale_change",
        "physical_crop.expanded_case_policy",
        errors,
    )
    model_input_pipeline_validated = _bool(
        physical.get("model_ready_3d_dce7_pipeline_validated"),
        "physical_crop.model_ready_3d_dce7_pipeline_validated",
        errors,
    )
    for field in (
        "scale_metadata_is_model_input",
        "mask_is_model_input",
        "bbox_or_target_geometry_is_model_input",
    ):
        _same_value(physical.get(field), False, f"physical_crop.{field}", errors)

    if detail is not None and context is not None:
        detail_fov = detail["fov_xyz_mm"]
        context_fov = context["fov_xyz_mm"]
        detail_spacing = detail["spacing_xyz_mm"]
        context_spacing = context["spacing_xyz_mm"]
        assert isinstance(detail_fov, list) and isinstance(context_fov, list)
        assert isinstance(detail_spacing, list) and isinstance(context_spacing, list)
        if any(outer + 1e-9 < inner for outer, inner in zip(context_fov, detail_fov, strict=True)):
            errors.append("C2 context FOV 必须逐轴包含 detail FOV")
        if not any(outer > inner + 1e-9 for outer, inner in zip(context_fov, detail_fov, strict=True)):
            errors.append("C2 context FOV 必须至少一轴严格大于 detail FOV")
        if any(coarse + 1e-9 < fine for coarse, fine in zip(context_spacing, detail_spacing, strict=True)):
            errors.append("C2 context spacing 不得比 detail spacing 更细")

    contracts = _mapping(config.get("contracts"), "contracts", errors)
    actual_contracts = set(contracts)
    expected_contracts = set(EXPECTED_CONTRACTS)
    if actual_contracts != expected_contracts:
        missing = sorted(expected_contracts - actual_contracts)
        unknown = sorted(actual_contracts - expected_contracts)
        if missing:
            errors.append(f"contracts 缺少：{missing}")
        if unknown:
            errors.append(f"contracts 含未知项：{unknown}")
    for contract_id, expected in EXPECTED_CONTRACTS.items():
        contract = _mapping(contracts.get(contract_id), f"contracts.{contract_id}", errors)
        for field, expected_value in expected.items():
            _same_value(
                contract.get(field),
                expected_value,
                f"contracts.{contract_id}.{field}",
                errors,
            )

    # Explicit relational checks make the scientific intent visible even if an
    # enum name is later edited accidentally.
    if contracts:
        for contract_id in ("C1A", "C1B", "C2A", "C2B"):
            contract = contracts.get(contract_id, {})
            if isinstance(contract, dict):
                if _bool(
                    contract.get("uses_future_support"),
                    f"contracts.{contract_id}.uses_future_support",
                    errors,
                ) is True:
                    errors.append(f"{contract_id} 不得使用 future support")
                if _bool(
                    contract.get("audit_only"),
                    f"contracts.{contract_id}.audit_only",
                    errors,
                ) is True:
                    errors.append(f"{contract_id} 是 deployable candidate，不能标 audit_only")

        c1c = contracts.get("C1C", {})
        if isinstance(c1c, dict):
            if c1c.get("audit_only") is not True:
                errors.append("C1C 必须 audit_only=true")
            if c1c.get("model_eligible") is not False:
                errors.append("C1C 必须 model_eligible=false")
            if c1c.get("uses_future_support") is not True:
                errors.append("C1C 必须显式 uses_future_support=true")

        for multiscale, detail_only in (("C2A", "C1A"), ("C2B", "C1B")):
            left = contracts.get(multiscale, {})
            right = contracts.get(detail_only, {})
            if isinstance(left, dict) and isinstance(right, dict):
                for field in ("anchor_policy", "reference_visits", "causal_deployability"):
                    if left.get(field) != right.get(field):
                        errors.append(
                            f"{multiscale}.{field} 必须与 {detail_only}.{field} 一致，"
                            "确保 detail/context 共中心"
                        )

    leakage = _mapping(config.get("leakage"), "leakage", errors)
    for field in (
        "mask_is_model_input",
        "crop_scale_metadata_is_model_input",
        "bbox_geometry_is_model_input",
    ):
        _same_value(leakage.get(field), False, f"leakage.{field}", errors)
    _same_value(
        leakage.get("training_loader_rejects_audit_only"),
        True,
        "leakage.training_loader_rejects_audit_only",
        errors,
    )
    _same_value(
        leakage.get("audit_metadata_storage"),
        "sidecar_only_not_loaded_by_model",
        "leakage.audit_metadata_storage",
        errors,
    )

    stage_b = _mapping(config.get("stage_b"), "stage_b", errors)
    _same_value(
        stage_b.get("conditional_on_stage_a_go"),
        True,
        "stage_b.conditional_on_stage_a_go",
        errors,
    )
    _same_value(stage_b.get("authorized_target"), "FTV_ONLY", "stage_b.authorized_target", errors)
    for field in (
        "ld_grounding_authorized",
        "pcr_supervision_authorized",
        "transition_modification_authorized",
    ):
        _same_value(stage_b.get(field), False, f"stage_b.{field}", errors)

    diagnostics: dict[str, Any] = {
        "schema_version": config.get("schema_version"),
        "contract_ids": sorted(actual_contracts),
        "detail_grid": detail,
        "context_grid": context,
        "c1c_audit_only": (
            contracts.get("C1C", {}).get("audit_only")
            if isinstance(contracts.get("C1C"), dict)
            else None
        ),
        "c1c_uses_future_support": (
            contracts.get("C1C", {}).get("uses_future_support")
            if isinstance(contracts.get("C1C"), dict)
            else None
        ),
        "model_geometry_inputs_disabled": all(
            leakage.get(field) is False
            for field in (
                "mask_is_model_input",
                "crop_scale_metadata_is_model_input",
                "bbox_geometry_is_model_input",
            )
        ),
        "image_only_rigid_registration_sensitivity_completed": (
            registration_sensitivity
        ),
        "model_ready_3d_dce7_pipeline_validated": (
            model_input_pipeline_validated
        ),
        "production_orientation_canonicalization_validated": (
            orientation_contract_validated
        ),
    }
    return errors, diagnostics


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "config": str(config_path), "errors": [str(exc)]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc

    errors, diagnostics = validate_config(payload)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "errors": errors,
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
