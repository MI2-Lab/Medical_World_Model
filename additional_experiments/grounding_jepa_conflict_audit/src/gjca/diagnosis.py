"""预注册 H1→H4 诊断判定机。

本模块不读取正式数据，也不计算统计量。它只接收上游已聚合且已完成
Holm 校正的 signals，机械执行冻结实验计划中的层级判定。

输入 schema:

    {
        "core_eligible": bool,
        "validation": {
            "D_cos": float, "D_neg": float, "D_mbase": float,
            "D_mfail": float, "D_ratio": float, "Q_ratio": float,
            "rho_cos": float | None, "rho_mbase": float | None,
            "rho_ratio": float | None,
        },
        "train": { ...同上... },
        "h1_holm_p": {
            "D_cos": float, "D_neg": float, "D_mbase": float,
            "rho_cos": float, "rho_mbase": float, "D_mfail": float,
        },
        "h2_holm_p": {"D_ratio": float, "rho_ratio": float},
        "dynamics": {
            endpoint: {"oriented_rho": float | None, "p_holm": float}
        },
    }

常量输入造成的不可用 rho 可为 None 或 non-finite；它不被伪装为零。
不可用 Holm p 按计划固定以 1.0 代入。
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any


SIGNAL_KEYS = (
    "D_cos",
    "D_neg",
    "D_mbase",
    "D_mfail",
    "D_ratio",
    "Q_ratio",
    "rho_cos",
    "rho_mbase",
    "rho_ratio",
)
H1_ENDPOINTS = (
    "D_cos",
    "D_neg",
    "D_mbase",
    "rho_cos",
    "rho_mbase",
    "D_mfail",
)
H2_ENDPOINTS = ("D_ratio", "rho_ratio")
DYNAMICS_ENDPOINTS = (
    "selected_epoch",
    "last_epoch",
    "last_minus_selected_epoch",
    "representation_collapse",
    "ftv_pressure",
    "cumulative_grounded_exposure",
    "post_selected_base_deterioration",
    "post_selected_ftv_improvement",
)

P_HOLM_THRESHOLD = 0.10
COSINE_DIFFERENCE_THRESHOLD = -0.10
FRACTION_DIFFERENCE_THRESHOLD = 0.10
MBASE_DIFFERENCE_THRESHOLD = -0.10
CORRELATION_MAGNITUDE_THRESHOLD = 0.35
MBASE_FAILURE_DIFFERENCE_THRESHOLD = 0.05
NORM_RATIO_QUOTIENT_THRESHOLD = 1.50
NORM_RATIO_DIFFERENCE_THRESHOLD = 0.25
DYNAMICS_ANCHOR_THRESHOLD = 0.50
DYNAMICS_SUPPORT_THRESHOLD = 0.35

RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "H1": ("PCGrad", "gradient normalization"),
    "H2": ("gradient normalization", "grounding warm-up"),
    "H3": ("checkpoint averaging", "grounding warm-up"),
    "H4": (
        "fixed batch order/composition stochastic replicate",
        "checkpoint averaging",
    ),
}
INVALID_LABEL = "UNDETERMINED_DATA_QUALITY"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 mapping")
    return value


def _number(value: Any, name: str) -> float | None:
    """返回有限数；None/NaN/Inf 保持为 unavailable。"""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} 不得以 boolean 充当数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} 必须是数值或 None") from error
    return result if math.isfinite(result) else None


def _signal_block(
    value: Any,
    name: str,
    quality_issues: list[str],
) -> dict[str, float | None]:
    block = _mapping(value, name)
    missing = [key for key in SIGNAL_KEYS if key not in block]
    if missing:
        raise KeyError(f"{name} 缺少 signal: {missing}")
    parsed = {key: _number(block[key], f"{name}.{key}") for key in SIGNAL_KEYS}
    # 这些组效应必须由有限 core run values 得到；相关可因常量输入而 unavailable。
    for key in ("D_cos", "D_neg", "D_mbase", "D_mfail", "D_ratio", "Q_ratio"):
        if parsed[key] is None:
            quality_issues.append(f"{name}.{key}:nonfinite")
    quotient = parsed["Q_ratio"]
    if quotient is not None and quotient <= 0:
        quality_issues.append(f"{name}.Q_ratio:nonpositive")
    return parsed


def _holm_p(value: Any, name: str) -> tuple[float, bool]:
    if isinstance(value, Mapping):
        item = _mapping(value, name)
        if "p_holm" not in item:
            raise KeyError(f"{name} mapping 缺少 p_holm")
        value = item["p_holm"]
    parsed = _number(value, name)
    if parsed is None:
        return 1.0, True
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} 必须在 [0,1]")
    return parsed, False


def _holm_family(
    value: Any,
    endpoints: tuple[str, ...],
    name: str,
) -> tuple[dict[str, float], list[str]]:
    family = _mapping(value, name)
    missing = [endpoint for endpoint in endpoints if endpoint not in family]
    if missing:
        raise KeyError(f"{name} 缺少预注册端点: {missing}")
    parsed: dict[str, float] = {}
    substituted: list[str] = []
    for endpoint in endpoints:
        parsed[endpoint], unavailable = _holm_p(family[endpoint], f"{name}.{endpoint}")
        if unavailable:
            substituted.append(endpoint)
    return parsed, substituted


def _dynamics(
    value: Any,
) -> tuple[dict[str, dict[str, float | None]], list[str]]:
    source = _mapping(value, "dynamics")
    missing = [endpoint for endpoint in DYNAMICS_ENDPOINTS if endpoint not in source]
    if missing:
        raise KeyError(f"dynamics 缺少预注册端点: {missing}")
    parsed: dict[str, dict[str, float | None]] = {}
    substituted: list[str] = []
    for endpoint in DYNAMICS_ENDPOINTS:
        item = _mapping(source[endpoint], f"dynamics.{endpoint}")
        if "oriented_rho" not in item or "p_holm" not in item:
            raise KeyError(f"dynamics.{endpoint} 必须含 oriented_rho/p_holm")
        p_value, unavailable = _holm_p(item["p_holm"], f"dynamics.{endpoint}.p_holm")
        parsed[endpoint] = {
            "oriented_rho": _number(
                item["oriented_rho"], f"dynamics.{endpoint}.oriented_rho"
            ),
            "p_holm": p_value,
        }
        if unavailable:
            substituted.append(endpoint)
    return parsed, substituted


def _le(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _abs_lt(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value) < threshold


def _h1(
    validation: Mapping[str, float | None],
    train: Mapping[str, float | None],
    holm_p: Mapping[str, float],
) -> dict[str, Any]:
    v_rho = {
        "rho_cos": _le(validation["rho_cos"], -CORRELATION_MAGNITUDE_THRESHOLD),
        "rho_mbase": _le(validation["rho_mbase"], -CORRELATION_MAGNITUDE_THRESHOLD),
    }
    v = {
        "V1_D_cos_le_minus_0.10": _le(validation["D_cos"], COSINE_DIFFERENCE_THRESHOLD),
        "V2_D_neg_ge_plus_0.10": _ge(
            validation["D_neg"], FRACTION_DIFFERENCE_THRESHOLD
        ),
        "V3_D_mbase_le_minus_0.10": _le(
            validation["D_mbase"], MBASE_DIFFERENCE_THRESHOLD
        ),
        "V4_negative_rho_ge_0.35": any(v_rho.values()),
        "V5_D_mfail_ge_plus_0.05": _ge(
            validation["D_mfail"], MBASE_FAILURE_DIFFERENCE_THRESHOLD
        ),
    }
    # T4 只能由 validation 中真正令 V4 越阈的同名 rho 支持。
    t_rho = {
        endpoint: crossed and _lt(train[endpoint], 0.0)
        for endpoint, crossed in v_rho.items()
    }
    t = {
        "T1_D_cos_lt_0": _lt(train["D_cos"], 0.0),
        "T2_D_neg_gt_0": _gt(train["D_neg"], 0.0),
        "T3_D_mbase_lt_0": _lt(train["D_mbase"], 0.0),
        "T4_same_negative_rho_lt_0": any(t_rho.values()),
        "T5_D_mfail_gt_0": _gt(train["D_mfail"], 0.0),
    }
    crossed = {
        "D_cos": v["V1_D_cos_le_minus_0.10"],
        "D_neg": v["V2_D_neg_ge_plus_0.10"],
        "D_mbase": v["V3_D_mbase_le_minus_0.10"],
        "rho_cos": v_rho["rho_cos"],
        "rho_mbase": v_rho["rho_mbase"],
        "D_mfail": v["V5_D_mfail_ge_plus_0.05"],
    }
    qualifying = [
        endpoint
        for endpoint in H1_ENDPOINTS
        if crossed[endpoint] and holm_p[endpoint] <= P_HOLM_THRESHOLD
    ]
    v_count = sum(v.values())
    t_count = sum(t.values())
    conditions = {
        "validation_at_least_3_of_5": v_count >= 3,
        "train_at_least_3_same_direction": t_count >= 3,
        "qualifying_practical_endpoint_p_holm_le_0.10": bool(qualifying),
    }
    return {
        "validation_signals": v,
        "validation_signal_count": v_count,
        "validation_rho_threshold_crossed": v_rho,
        "train_directional_signals": t,
        "train_directional_signal_count": t_count,
        "train_same_rho_replication": t_rho,
        "practical_crossed_by_endpoint": crossed,
        "holm_p": dict(holm_p),
        "qualifying_endpoints": qualifying,
        "conditions": conditions,
        "all_conditions_met": all(conditions.values()),
    }


def _h2(
    validation: Mapping[str, float | None],
    train: Mapping[str, float | None],
    holm_p: Mapping[str, float],
    h1_signal_count: int,
) -> dict[str, Any]:
    q_crossed = _ge(validation["Q_ratio"], NORM_RATIO_QUOTIENT_THRESHOLD)
    d_crossed = _ge(validation["D_ratio"], NORM_RATIO_DIFFERENCE_THRESHOLD)
    scale = q_crossed or d_crossed
    rho_crossed = _ge(validation["rho_ratio"], CORRELATION_MAGNITUDE_THRESHOLD)
    crossed = {"D_ratio": scale, "rho_ratio": rho_crossed}
    qualifying = [
        endpoint
        for endpoint in H2_ENDPOINTS
        if crossed[endpoint] and holm_p[endpoint] <= P_HOLM_THRESHOLD
    ]
    conditions = {
        "h1_primary_signal_count_lt_3": h1_signal_count < 3,
        "validation_scale_practical": scale,
        "validation_rho_ratio_ge_0.35": rho_crossed,
        "train_Q_gt_1_and_D_gt_0": (
            _gt(train["Q_ratio"], 1.0) and _gt(train["D_ratio"], 0.0)
        ),
        "train_rho_ratio_gt_0": _gt(train["rho_ratio"], 0.0),
        "qualifying_practical_endpoint_p_holm_le_0.10": bool(qualifying),
    }
    return {
        "validation_scale_components": {
            "Q_ratio_ge_1.50": q_crossed,
            "D_ratio_ge_0.25": d_crossed,
        },
        "practical_crossed_by_endpoint": crossed,
        "holm_p": dict(holm_p),
        "qualifying_endpoints": qualifying,
        "conditions": conditions,
        "all_conditions_met": all(conditions.values()),
    }


def _h3(
    validation: Mapping[str, float | None],
    dynamics: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    anchors = [
        endpoint
        for endpoint in DYNAMICS_ENDPOINTS
        if _ge(dynamics[endpoint]["oriented_rho"], DYNAMICS_ANCHOR_THRESHOLD)
        and _le(dynamics[endpoint]["p_holm"], P_HOLM_THRESHOLD)
    ]
    supports = [
        endpoint
        for endpoint in DYNAMICS_ENDPOINTS
        if _ge(dynamics[endpoint]["oriented_rho"], DYNAMICS_SUPPORT_THRESHOLD)
    ]
    pairs = [
        [anchor, support]
        for anchor in anchors
        for support in supports
        if anchor != support
    ]
    quotient = validation["Q_ratio"]
    symmetric_q = (
        max(quotient, 1.0 / quotient) if quotient is not None and quotient > 0 else None
    )
    small = {
        "abs_D_cos_lt_0.10": _abs_lt(validation["D_cos"], 0.10),
        "abs_D_neg_lt_0.10": _abs_lt(validation["D_neg"], 0.10),
        "abs_D_mbase_lt_0.10": _abs_lt(validation["D_mbase"], 0.10),
        "abs_rho_cos_lt_0.35": _abs_lt(validation["rho_cos"], 0.35),
        "abs_rho_mbase_lt_0.35": _abs_lt(validation["rho_mbase"], 0.35),
        "abs_D_mfail_lt_0.05": _abs_lt(validation["D_mfail"], 0.05),
        "symmetric_Q_ratio_lt_1.50": _lt(symmetric_q, 1.50),
        "abs_D_ratio_lt_0.25": _abs_lt(validation["D_ratio"], 0.25),
        "abs_rho_ratio_lt_0.35": _abs_lt(validation["rho_ratio"], 0.35),
    }
    conditions = {
        "anchor_exists": bool(anchors),
        "distinct_support_exists": bool(pairs),
        "all_nine_gradient_signals_strictly_small": all(small.values()),
    }
    return {
        "dynamics": {
            endpoint: dict(dynamics[endpoint]) for endpoint in DYNAMICS_ENDPOINTS
        },
        "anchor_endpoints": anchors,
        "support_endpoints": supports,
        "distinct_anchor_support_pairs": pairs,
        "symmetric_Q_ratio": symmetric_q,
        "strict_small_conditions": small,
        "strict_small_count": sum(small.values()),
        "conditions": conditions,
        "all_conditions_met": all(conditions.values()),
    }


def evaluate_hypotheses(signals: Mapping[str, Any]) -> dict[str, Any]:
    """机械执行 H1→H2→H3→H4/UNDETERMINED。

    正常结果恰有四个 H1–H4 decision rows，且恰一行 selected。若 core
    不合格，四行均 selected=False，顶层结果为 UNDETERMINED_DATA_QUALITY；
    此时绝不误贴 H4。
    """

    payload = _mapping(signals, "signals")
    if "core_eligible" not in payload or not isinstance(payload["core_eligible"], bool):
        raise TypeError("signals.core_eligible 必须是 boolean")
    required = ("validation", "train", "h1_holm_p", "h2_holm_p", "dynamics")
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"signals 缺少必要字段: {missing}")

    quality_issues: list[str] = []
    validation = _signal_block(payload["validation"], "validation", quality_issues)
    train = _signal_block(payload["train"], "train", quality_issues)
    h1_p, h1_substituted = _holm_family(payload["h1_holm_p"], H1_ENDPOINTS, "h1_holm_p")
    h2_p, h2_substituted = _holm_family(payload["h2_holm_p"], H2_ENDPOINTS, "h2_holm_p")
    dynamics, h3_substituted = _dynamics(payload["dynamics"])

    declared_core = payload["core_eligible"]
    core = declared_core and not quality_issues

    h1 = _h1(validation, train, h1_p)
    h1_rule = core and h1["all_conditions_met"]

    h2 = _h2(
        validation,
        train,
        h2_p,
        int(h1["validation_signal_count"]),
    )
    h2_reached = core and not h1_rule
    h2_eligible = h2_reached and h1["validation_signal_count"] < 3
    h2_rule = h2_eligible and h2["all_conditions_met"]

    h3 = _h3(validation, dynamics)
    h3_reached = core and not h1_rule and not h2_rule
    h3_rule = h3_reached and h3["all_conditions_met"]

    h4_reached = core and not h1_rule and not h2_rule and not h3_rule
    h4_rule = h4_reached

    if not core:
        selected_hypothesis = INVALID_LABEL
    elif h1_rule:
        selected_hypothesis = "H1"
    elif h2_rule:
        selected_hypothesis = "H2"
    elif h3_rule:
        selected_hypothesis = "H3"
    else:
        selected_hypothesis = "H4"

    details: dict[str, Any] = {
        "H1": {
            **h1,
            "hierarchy_reached": core,
            "rule_eligible": core,
            "rule_satisfied": h1_rule,
        },
        "H2": {
            **h2,
            "hierarchy_reached": h2_reached,
            "rule_eligible": h2_eligible,
            "rule_satisfied": h2_rule,
        },
        "H3": {
            **h3,
            "hierarchy_reached": h3_reached,
            "rule_eligible": h3_reached,
            "rule_satisfied": h3_rule,
        },
        "H4": {
            "conditions": {
                "core_eligible": core,
                "H1_not_satisfied": not h1_rule,
                "H2_not_satisfied": not h2_rule,
                "H3_not_satisfied": not h3_rule,
            },
            "hierarchy_reached": h4_reached,
            "rule_eligible": h4_reached,
            "rule_satisfied": h4_rule,
        },
    }

    rows: list[dict[str, Any]] = []
    for hypothesis in ("H1", "H2", "H3", "H4"):
        first, second = RECOMMENDATIONS[hypothesis]
        selected = selected_hypothesis == hypothesis
        rows.append(
            {
                "hypothesis": hypothesis,
                "hierarchy_reached": bool(details[hypothesis]["hierarchy_reached"]),
                "rule_eligible": bool(details[hypothesis]["rule_eligible"]),
                "rule_satisfied": bool(details[hypothesis]["rule_satisfied"]),
                "selected": selected,
                "decision_status": (
                    INVALID_LABEL
                    if selected_hypothesis == INVALID_LABEL
                    else ("SELECTED" if selected else "NOT_SELECTED")
                ),
                "selected_hypothesis": selected_hypothesis,
                "first_recommendation": first,
                "second_recommendation": second,
                "condition_details_json": json.dumps(
                    details[hypothesis],
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            }
        )

    selected_count = sum(bool(row["selected"]) for row in rows)
    expected_count = 0 if selected_hypothesis == INVALID_LABEL else 1
    if len(rows) != 4 or selected_count != expected_count:
        raise AssertionError("hypothesis decision 唯一性合同破坏")

    chosen = RECOMMENDATIONS.get(selected_hypothesis)
    return {
        "schema_version": 1,
        "declared_core_eligible": declared_core,
        "core_eligible": core,
        "data_quality_issues": quality_issues,
        "holm_p_substituted_as_1": {
            "H1": h1_substituted,
            "H2": h2_substituted,
            "H3": h3_substituted,
        },
        "selected_hypothesis": selected_hypothesis,
        "first_recommendation": chosen[0] if chosen is not None else None,
        "second_recommendation": chosen[1] if chosen is not None else None,
        "decision_rows": rows,
        "condition_details": details,
    }


def _neutral_payload() -> dict[str, Any]:
    block = {
        "D_cos": 0.0,
        "D_neg": 0.0,
        "D_mbase": 0.0,
        "D_mfail": 0.0,
        "D_ratio": 0.0,
        "Q_ratio": 1.0,
        "rho_cos": 0.0,
        "rho_mbase": 0.0,
        "rho_ratio": 0.0,
    }
    return {
        "core_eligible": True,
        "validation": dict(block),
        "train": dict(block),
        "h1_holm_p": {endpoint: 1.0 for endpoint in H1_ENDPOINTS},
        "h2_holm_p": {endpoint: 1.0 for endpoint in H2_ENDPOINTS},
        "dynamics": {
            endpoint: {"oriented_rho": 0.0, "p_holm": 1.0}
            for endpoint in DYNAMICS_ENDPOINTS
        },
    }


def synthetic_self_test() -> dict[str, Any]:
    """覆盖 H1/H2/H3/H4、invalid 与严格/非严格边界。"""

    h1_input = _neutral_payload()
    h1_input["validation"].update({"D_cos": -0.10, "D_neg": 0.10, "D_mbase": -0.10})
    h1_input["train"].update({"D_cos": -0.01, "D_neg": 0.01, "D_mbase": -0.01})
    h1_input["h1_holm_p"]["D_cos"] = 0.10
    h1 = evaluate_hypotheses(h1_input)
    assert h1["selected_hypothesis"] == "H1"
    assert h1["condition_details"]["H1"]["validation_signal_count"] == 3

    h2_input = _neutral_payload()
    h2_input["validation"].update({"D_ratio": 0.25, "Q_ratio": 1.49, "rho_ratio": 0.35})
    h2_input["train"].update({"D_ratio": 0.01, "Q_ratio": 1.01, "rho_ratio": 0.01})
    h2_input["h2_holm_p"]["D_ratio"] = 0.10
    h2 = evaluate_hypotheses(h2_input)
    assert h2["selected_hypothesis"] == "H2"

    h3_input = _neutral_payload()
    h3_input["dynamics"]["selected_epoch"] = {
        "oriented_rho": 0.50,
        "p_holm": 0.10,
    }
    h3_input["dynamics"]["last_epoch"] = {
        "oriented_rho": 0.35,
        "p_holm": 1.0,
    }
    h3 = evaluate_hypotheses(h3_input)
    assert h3["selected_hypothesis"] == "H3"
    assert h3["condition_details"]["H3"]["distinct_anchor_support_pairs"]

    h4 = evaluate_hypotheses(_neutral_payload())
    assert h4["selected_hypothesis"] == "H4"

    invalid_input = copy.deepcopy(h1_input)
    invalid_input["core_eligible"] = False
    invalid = evaluate_hypotheses(invalid_input)
    assert invalid["selected_hypothesis"] == INVALID_LABEL
    assert sum(row["selected"] for row in invalid["decision_rows"]) == 0
    assert not invalid["condition_details"]["H4"]["rule_satisfied"]

    # H3 的 nine-small 全部为严格不等式：恰等于阈值不算 small。
    boundary_input = copy.deepcopy(h3_input)
    boundary_input["validation"]["D_cos"] = -0.10
    boundary = evaluate_hypotheses(boundary_input)
    assert boundary["selected_hypothesis"] == "H4"
    assert not boundary["condition_details"]["H3"]["strict_small_conditions"][
        "abs_D_cos_lt_0.10"
    ]

    reciprocal_input = copy.deepcopy(h3_input)
    reciprocal_input["validation"]["Q_ratio"] = 2.0 / 3.0
    reciprocal = evaluate_hypotheses(reciprocal_input)
    assert reciprocal["selected_hypothesis"] == "H4"
    assert not reciprocal["condition_details"]["H3"]["strict_small_conditions"][
        "symmetric_Q_ratio_lt_1.50"
    ]

    # 显著但未越过 practical threshold 的端点不得替代 qualifying p。
    nonqualifying_input = copy.deepcopy(h1_input)
    nonqualifying_input["h1_holm_p"] = {endpoint: 1.0 for endpoint in H1_ENDPOINTS}
    nonqualifying_input["h1_holm_p"]["rho_cos"] = 0.001
    nonqualifying = evaluate_hypotheses(nonqualifying_input)
    assert nonqualifying["selected_hypothesis"] == "H4"
    assert not nonqualifying["condition_details"]["H1"]["qualifying_endpoints"]

    cases = {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "invalid": invalid,
        "boundary_strict_small": boundary,
        "boundary_reciprocal_ratio": reciprocal,
        "nonqualifying_holm": nonqualifying,
    }
    for name, result in cases.items():
        assert len(result["decision_rows"]) == 4, name
        selected_count = sum(bool(row["selected"]) for row in result["decision_rows"])
        assert selected_count == (0 if name == "invalid" else 1), name
    return {
        "status": "ok",
        "cases": {
            name: result["selected_hypothesis"] for name, result in cases.items()
        },
        "decision_rows_per_case": 4,
        "normal_selected_count": 1,
        "invalid_selected_count": 0,
    }


if __name__ == "__main__":
    print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
