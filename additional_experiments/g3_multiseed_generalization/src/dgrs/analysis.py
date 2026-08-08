"""G3 multi-seed generalization 的正式聚合、统计、绘图与机械判定。

本模块只读取 finalized checkpoint/feature、训练选择及 outer-test 患者级
prediction。它不会重拟合 probe/readout，不会选择 checkpoint/lambda，也不会
把患者级行写入公开 ``metrics/final``。正式主 uncertainty 是五个独立训练 seed
上的 t-CI；patient bootstrap 只条件于已拟合 seed，crossed bootstrap 仅作支持性
敏感性分析。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import font_manager
from matplotlib.figure import Figure
from PIL import Image
from scipy.special import expit
from scipy.stats import beta, binomtest, pearsonr, rankdata, spearmanr, t as student_t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .features import (
    extraction_implementation_sha256,
    validate_feature_against_canonical,
)
from .pcr import (
    C_GRID as PCR_C_GRID,
    FEATURE_DIMS as PCR_FEATURE_DIMS,
    FEATURE_SCHEMAS as PCR_FEATURE_SCHEMAS,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    LOGISTIC_TOL,
    PCR_READOUT_SEED,
    PENALTIES as PCR_PENALTIES,
    PREDICTION_COLUMNS as PCR_PREDICTION_COLUMNS,
    SELECTION_COLUMNS as PCR_SELECTION_COLUMNS,
    pcr_implementation_sha256,
)
from .probes import (
    ALPHAS as PROBE_ALPHAS,
    PREDICTION_COLUMNS as PROBE_PREDICTION_COLUMNS,
    RIDGE_MAX_ITER,
    RIDGE_SOLVER,
    RIDGE_TOL,
    SELECTION_COLUMNS as PROBE_SELECTION_COLUMNS,
    probe_implementation_sha256,
)
from .targets import patient_hash
from .training import implementation_sha256 as training_implementation_sha256


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCHEMA_VERSION = 1
SEEDS = (2026, 3026, 4026, 5026, 6026)
MODELS = ("G1", "G3")
FOLDS = tuple(range(5))
TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
REGRESSION_METRICS = (
    "spearman",
    "pearson",
    "r2",
    "mae",
    "rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
)
CLASSIFICATION_METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
EXPECTED_PROBE_ROWS = 26_250
EXPECTED_PCR_ROWS = 24_240
EXPECTED_ANALYSIS_SEED = 20260807
EXPECTED_CONDITIONAL_REPLICATES = 2000
EXPECTED_CROSSED_REPLICATES = 5000
EXPECTED_SOURCE_COMMIT = "596d6d509aaf62c5385344a83b8ed66dd301ee79"
EXPECTED_BRANCH = "feature/ispy-clean-corejepa"
EXPECTED_PLAN_SHA256 = (
    "394402aa8235b26f07b98a32426639a915bad80c53fc49cb053e7123e97ad06c"
)
EXPECTED_PLAN_FREEZE_SHA256 = (
    "7e4cb0ea26fce8f192a0e75b26365e13876dfe8b01a9f6d5f261efa9fb273dfc"
)
EXPECTED_SOURCE_FREEZE_SHA256 = (
    "5036c77f1d73bafedac16bb9837e38572cfd02d61a1506cd989506b836cdd05b"
)
EXPECTED_SOURCE_IMPLEMENTATION_SHA256 = (
    "9314ab06c47c5d126c4687865b4dc5f92816c7dcdb3b372abdb313b30ffe6bdf"
)
EXPECTED_TRAINING_IMPLEMENTATION_SHA256 = (
    "41ba62f3d12fa52e0c4f5290d7b19cca34032b1fc8ae5113bd1e353245f355ab"
)
EXPECTED_PREFLIGHT_SHA256 = (
    "3fcce3e48f2b5fa1a6cb5b1fc2ab56392360e25636cb3e9d5f3eccc27d9c43df"
)
EXPECTED_SMOKE_GATE_SHA256 = (
    "1c33dcb047d5f42737fcc5438e1617ea13570d529c085c70dae93fae0093296b"
)
EXPECTED_RAW_TARGET_FILE_SHA256 = (
    "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
)
EXPECTED_RAW_FTV_SHA256 = (
    "41b419f6e098dade710ee8963ccd6245ce3ea9bd32687afbef1223cafde9529b"
)
EXPECTED_PROBE_RAW_TARGETS_SHA256 = (
    "512749ccf986de4af4c0109b4ce060c61a90112816895a2ae7423784ea60de4e"
)
EXPECTED_FTV_TRANSFORM_SHA256 = {
    0: "8df48a908a5d56f76a2dd1a5f52b7189b03ce64e60743f856ef14afca07ebd5b",
    1: "6b582c2bb22e8208bc2e149eec032d179182fde212b94bcf6161bd274b38b4d4",
    2: "fcdf72ea26da1ff49efbdc937c78761e41d54640dae20289ac73a193e9cee23a",
    3: "a666b556e87c955214869547c6d54f083b8f975838c12461cc1158332532792c",
    4: "cb207a387900cc9ebc3deb7dca8e448bdbea083aae495af07fd11200008d6a9c",
}
EXPECTED_STATIC_TRANSFORM_SHA256 = {
    0: "22b169836480748189b7301010e75d64672e3553db51f1a4d22ab6b03ac3e5e3",
    1: "165a81ec0860d6fb3b5eb4982ec534fd260693231f4154b8abe60bd23ef0226c",
    2: "b2cfdd8e77db596c5c14750550cd0d468d9438130aca8d16eec22b65b8284a91",
    3: "1621665ea513536f28011f9ace1b4eda2252efe6de019ac38ef0f76b22a3c9cb",
    4: "e7e87975ce069383dd0627a8c74b2ab24638a3c66aac642e5ce3c9d20d51d278",
}
EXPECTED_CHANGE_TRANSFORM_SHA256 = {
    0: "7b7ab13800b8a296f70f5cca7a8abde34c27f2adb16d1cd7e0ed3d0ed9780e37",
    1: "d3d89808922b70dc27f99ce69f3328211b8b75c67c662636858b1bbe3b4d0ed4",
    2: "7c768b09f5cda5085790050e95f9d718b4882e939592395470d637a5a8d56009",
    3: "681a2618240f9b57345b625a20f707a4f58d1dd2e7197418b9ee081161f4b5ce",
    4: "60f804451765b13a7d91c0c2b69b285390d1b2a32f2cd2c216ff9e13ecc1837c",
}
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_VALUE = re.compile(r"^[0-9a-f]{64}$")

PROBE_FALSE_FLAGS = (
    "test_used_for_target_transform",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
)
PCR_FALSE_FLAGS = (
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_hyperparameter_selection",
    "test_used_for_threshold_selection",
)


class AnalysisInputError(RuntimeError):
    """输入缺失、歧义或违反冻结计划。"""


@dataclass(frozen=True)
class AnalysisConfig:
    """正式聚合配置；正式值由 CLI 锁定。"""

    prediction_root: Path = EXPERIMENT_ROOT / "predictions"
    metric_input_root: Path = EXPERIMENT_ROOT / "metrics"
    history_root: Path = EXPERIMENT_ROOT / "metrics" / "training" / "formal"
    checkpoint_root: Path = EXPERIMENT_ROOT / "checkpoints" / "formal"
    feature_root: Path = EXPERIMENT_ROOT / "features"
    metric_dir: Path = EXPERIMENT_ROOT / "metrics" / "final"
    figure_dir: Path = EXPERIMENT_ROOT / "figures" / "final"
    report_path: Path = EXPERIMENT_ROOT / "reports" / "final_report.md"
    conditional_replicates: int = EXPECTED_CONDITIONAL_REPLICATES
    crossed_replicates: int = EXPECTED_CROSSED_REPLICATES
    seed: int = EXPECTED_ANALYSIS_SEED
    overwrite: bool = False
    audit_checkpoints: bool = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *map(str, parts)], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _portable(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        token = hashlib.sha256(str(resolved.parent).encode()).hexdigest()[:12]
        return f"<external:{token}>/{resolved.name}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _strict_bool(series: pd.Series, label: str) -> pd.Series:
    mapping: dict[Any, bool] = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
    }
    result = series.map(mapping)
    if result.isna().any():
        examples = series.loc[result.isna()].astype(str).head(3).tolist()
        raise AnalysisInputError(f"{label} 含非法布尔值: {examples}")
    return result.astype(bool)


def _finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AnalysisInputError(f"{label}.{column} 含 NaN/Inf 或非数值")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise AnalysisInputError(f"{label} 缺列: {missing}")


def _require_sha(series: pd.Series, label: str) -> None:
    values = series.astype(str).str.lower().str.strip()
    if not values.map(lambda value: bool(SHA256_VALUE.fullmatch(value))).all():
        raise AnalysisInputError(f"{label} 必须逐行为 64 位 SHA-256")


def _require_exact_columns(
    frame: pd.DataFrame, expected: Sequence[str], label: str
) -> None:
    actual = tuple(map(str, frame.columns))
    expected_tuple = tuple(map(str, expected))
    if actual != expected_tuple:
        missing = sorted(set(expected_tuple).difference(actual))
        extra = sorted(set(actual).difference(expected_tuple))
        raise AnalysisInputError(
            f"{label} column schema 漂移: 缺={missing} 多={extra} 顺序一致={actual == expected_tuple}"
        )


def _numeric(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AnalysisInputError(f"{label} 含 NaN/Inf 或非数值")
    return values


def _integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = _numeric(series, label)
    if not np.equal(values, np.floor(values)).all():
        raise AnalysisInputError(f"{label} 必须为精确整数，禁止静默截断")
    return values.astype(np.int64)


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"{label} 非合法 JSON") from exc
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{label} JSON 顶层必须为 object")
    return payload


def _json_finite_vector(value: Any, label: str, expected_size: int) -> np.ndarray:
    try:
        payload = json.loads(str(value))
        vector = np.asarray(payload, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"{label} 非合法数值 JSON vector") from exc
    if vector.shape != (expected_size,) or not np.isfinite(vector).all():
        raise AnalysisInputError(
            f"{label} 必须是长度 {expected_size} 的 finite JSON vector"
        )
    return vector


def _validate_freeze_provenance() -> dict[str, Any]:
    """独立闭合 plan/source freeze；训练 checkpoint 再逐格绑定此结果。"""

    plan_path = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
    plan_freeze_path = EXPERIMENT_ROOT / "PLAN_FREEZE.json"
    source_freeze_path = EXPERIMENT_ROOT / "SOURCE_FREEZE.json"
    preflight_path = EXPERIMENT_ROOT / "metrics" / "preflight.json"
    smoke_gate_path = EXPERIMENT_ROOT / "metrics" / "smoke_gate.json"
    plan_freeze = _read_json(plan_freeze_path, "PLAN_FREEZE")
    source_freeze = _read_json(source_freeze_path, "SOURCE_FREEZE")
    preflight = _read_json(preflight_path, "preflight")
    smoke_gate = _read_json(smoke_gate_path, "smoke_gate")
    plan_sha = file_sha256(plan_path)
    expected_plan_file = str(plan_path.resolve().relative_to(REPO_ROOT.resolve()))
    frozen_artifacts = {
        plan_path: EXPECTED_PLAN_SHA256,
        plan_freeze_path: EXPECTED_PLAN_FREEZE_SHA256,
        source_freeze_path: EXPECTED_SOURCE_FREEZE_SHA256,
        preflight_path: EXPECTED_PREFLIGHT_SHA256,
        smoke_gate_path: EXPECTED_SMOKE_GATE_SHA256,
    }
    for path, expected_sha in frozen_artifacts.items():
        observed_sha = file_sha256(path)
        if observed_sha != expected_sha:
            raise AnalysisInputError(
                f"冻结证据 SHA 漂移: {_portable(path)}: {observed_sha} != {expected_sha}"
            )
    if (
        int(plan_freeze.get("schema_version", -1)) != 1
        or plan_freeze.get("status") != "frozen"
        or plan_freeze.get("frozen_before_formal_training") is not True
        or plan_freeze.get("formal_training_started") is not False
        or str(plan_freeze.get("plan_file")) != expected_plan_file
        or str(plan_freeze.get("plan_sha256")) != EXPECTED_PLAN_SHA256
        or plan_sha != EXPECTED_PLAN_SHA256
    ):
        raise AnalysisInputError("PLAN_FREEZE 与当前冻结计划不闭合")
    if (
        int(source_freeze.get("schema_version", -1)) != 1
        or source_freeze.get("status") != "frozen before formal training"
        or source_freeze.get("scope") != "formal_training_only"
        or source_freeze.get("formal_training_started_at_freeze") is not False
        or source_freeze.get("branch") != EXPECTED_BRANCH
        or source_freeze.get("start_commit") != EXPECTED_SOURCE_COMMIT
        or source_freeze.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or source_freeze.get("implementation_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_SHA256
    ):
        raise AnalysisInputError("SOURCE_FREEZE identity/plan/branch contract 漂移")
    files = source_freeze.get("source_files")
    if (
        not isinstance(files, dict)
        or int(source_freeze.get("source_file_count", -1)) != len(files)
        or len(files) != 25
    ):
        raise AnalysisInputError("SOURCE_FREEZE source_file_count/schema 非法")
    digest = hashlib.sha256()
    for relative, expected_sha in sorted(files.items()):
        path = (REPO_ROOT / str(relative)).resolve()
        if REPO_ROOT.resolve() not in path.parents or not path.is_file():
            raise AnalysisInputError(f"SOURCE_FREEZE 路径非法或缺失: {relative}")
        actual_sha = file_sha256(path)
        if actual_sha != str(expected_sha):
            raise AnalysisInputError(
                f"SOURCE_FREEZE live hash 漂移: {relative}: {actual_sha} != {expected_sha}"
            )
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(actual_sha.encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != EXPECTED_SOURCE_IMPLEMENTATION_SHA256:
        raise AnalysisInputError("SOURCE_FREEZE implementation manifest digest 不闭合")

    expected_seed_configs = {
        str(seed): {
            "seed_base": seed,
            "effective_seeds": [seed + fold for fold in FOLDS],
        }
        for seed in SEEDS
    }
    expected_transform_counts = {
        "0": (525, 247),
        "1": (525, 239),
        "2": (525, 240),
        "3": (526, 242),
        "4": (526, 225),
    }
    transform_rows = preflight.get("ftv_transforms")
    preflight_identity_ok = (
        int(preflight.get("schema_version", -1)) == 1
        and preflight.get("status") == "pass"
        and preflight.get("issues") == []
        and preflight.get("branch") == EXPECTED_BRANCH
        and preflight.get("head_at_preflight") == EXPECTED_SOURCE_COMMIT
        and preflight.get("plan_sha256") == EXPECTED_PLAN_SHA256
        and preflight.get("implementation_sha256")
        == EXPECTED_SOURCE_IMPLEMENTATION_SHA256
        and preflight.get("fold_manifest_sha256") == EXPECTED_FOLD_MANIFEST_SHA256
        and preflight.get("raw_target_file_sha256") == EXPECTED_RAW_TARGET_FILE_SHA256
        and preflight.get("seed_configs") == expected_seed_configs
        and preflight.get("formal_models") == list(MODELS)
        and preflight.get("forbidden_models") == ["G0", "G2", "G4"]
        and preflight.get("formal_checkpoint_count_at_preflight") == 50
        and preflight.get("cache_present") is True
        and float(preflight.get("g3_lambda_ftv", math.nan)) == 0.25
        and isinstance(transform_rows, dict)
        and set(transform_rows) == set(expected_transform_counts)
    )
    if not preflight_identity_ok:
        raise AnalysisInputError("preflight 冻结协议/数据/源码证据漂移")
    for fold_text, (train_count, paired_count) in expected_transform_counts.items():
        row = transform_rows[fold_text]
        fold = int(fold_text)
        transform_path = EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
        if (
            not isinstance(row, dict)
            or row.get("sha256") != EXPECTED_FTV_TRANSFORM_SHA256[fold]
            or row.get("train_patient_count") != train_count
            or row.get("paired_train_patient_count") != paired_count
            or file_sha256(transform_path) != EXPECTED_FTV_TRANSFORM_SHA256[fold]
        ):
            raise AnalysisInputError(f"preflight fold={fold} FTV transform 证据漂移")

    current_training_sha = training_implementation_sha256()
    if current_training_sha != EXPECTED_TRAINING_IMPLEMENTATION_SHA256:
        raise AnalysisInputError(
            "当前 training implementation 与 smoke/formal 冻结实现不一致"
        )
    if (
        int(smoke_gate.get("schema_version", -1)) != 1
        or smoke_gate.get("status") != "pass"
        or smoke_gate.get("issues") != []
        or smoke_gate.get("fold") != 3
        or smoke_gate.get("seeds") != [2026, 3026]
        or smoke_gate.get("models") != list(MODELS)
        or smoke_gate.get("run_count") != 4
        or smoke_gate.get("formal_checkpoint_count_at_gate") != 0
        or smoke_gate.get("test_used_for_selection") is not False
    ):
        raise AnalysisInputError("smoke gate identity/selection-leakage 证据漂移")
    smoke_runs = smoke_gate.get("runs")
    if not isinstance(smoke_runs, list) or len(smoke_runs) != 4:
        raise AnalysisInputError("smoke gate run grid 非 2-seed × 2-model")
    observed_smoke: dict[tuple[int, str], Mapping[str, Any]] = {}
    for run in smoke_runs:
        if not isinstance(run, dict):
            raise AnalysisInputError("smoke gate run schema 非 object")
        seed = int(run.get("seed_base", -1))
        model = str(run.get("model", "")).upper()
        key = (seed, model)
        if key in observed_smoke:
            raise AnalysisInputError(f"smoke gate run 重复: {key}")
        if (
            seed not in (2026, 3026)
            or model not in MODELS
            or int(run.get("fold", -1)) != 3
            or int(run.get("effective_seed", -1)) != seed + 3
            or run.get("implementation_sha256")
            != EXPECTED_TRAINING_IMPLEMENTATION_SHA256
            or run.get("ftv_transform_sha256") != EXPECTED_FTV_TRANSFORM_SHA256[3]
            or run.get("selection_mode") not in {"primary", "fallback"}
        ):
            raise AnalysisInputError(f"smoke gate run contract 漂移: {key}")
        checkpoint = (REPO_ROOT / str(run.get("checkpoint", ""))).resolve()
        if (
            REPO_ROOT.resolve() not in checkpoint.parents
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != run.get("checkpoint_sha256")
        ):
            raise AnalysisInputError(f"smoke checkpoint live hash 不闭合: {key}")
        selection = checkpoint.parent / "selection.json"
        history = (
            EXPERIMENT_ROOT
            / "metrics"
            / "training"
            / "smoke_multiseed"
            / f"seed_{seed}"
            / model.lower()
            / "fold_3.csv"
        )
        if (
            not selection.is_file()
            or file_sha256(selection) != run.get("selection_sha256")
            or not history.is_file()
            or file_sha256(history) != run.get("history_sha256")
        ):
            raise AnalysisInputError(f"smoke selection/history live hash 不闭合: {key}")
        observed_smoke[key] = run
    if set(observed_smoke) != {
        (seed, model) for seed in (2026, 3026) for model in MODELS
    }:
        raise AnalysisInputError("smoke gate run grid 缺格或多格")
    for seed in (2026, 3026):
        g1 = observed_smoke[(seed, "G1")]
        g3 = observed_smoke[(seed, "G3")]
        if g1.get("shared_initialization_sha256") != g3.get(
            "shared_initialization_sha256"
        ) or g1.get("split_hashes") != g3.get("split_hashes"):
            raise AnalysisInputError(f"smoke G1/G3 配对不闭合: seed={seed}")
    if observed_smoke[(2026, "G1")].get(
        "shared_initialization_sha256"
    ) == observed_smoke[(3026, "G1")].get("shared_initialization_sha256"):
        raise AnalysisInputError("smoke 不同 seed 共享初始化，随机性审计失败")
    return {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "plan_freeze_sha256": EXPECTED_PLAN_FREEZE_SHA256,
        "source_freeze_sha256": EXPECTED_SOURCE_FREEZE_SHA256,
        "source_implementation_sha256": EXPECTED_SOURCE_IMPLEMENTATION_SHA256,
        "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "smoke_gate_sha256": EXPECTED_SMOKE_GATE_SHA256,
        "training_implementation_sha256": current_training_sha,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }


def _normalise_transition(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    return text.replace("->", "→").replace("–", "→").replace("—", "→").replace("-", "→")


def _normalise_decision(value: Any) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("→", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"无法读取 {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{label} 顶层必须是 object: {path}")
    return payload


def _manifest_row(kind: str, path: Path, **extra: Any) -> dict[str, Any]:
    row = {
        "kind": kind,
        "path": _portable(path),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
    }
    row.update(extra)
    return row


def _expected_grid() -> set[tuple[int, str, int]]:
    return {(seed, model, fold) for seed in SEEDS for model in MODELS for fold in FOLDS}


def _prediction_path(root: Path, kind: str, seed: int, model: str, fold: int) -> Path:
    return (
        root / kind / f"seed_{seed}" / model / f"fold_{fold}" / "test_predictions.csv"
    )


def _selection_path(root: Path, kind: str, seed: int, model: str, fold: int) -> Path:
    return (
        root / kind / f"seed_{seed}" / model / f"fold_{fold}" / "selection_records.csv"
    )


def _summary_path(root: Path, kind: str, seed: int, model: str, fold: int) -> Path:
    return root / kind / f"seed_{seed}" / model / f"fold_{fold}" / "summary.json"


def _cell_order(task: str, cell: str) -> int:
    return (TIMEPOINTS if task == "static" else TRANSITIONS).index(cell)


def _normalise_probe_file(path: Path, seed: int, model: str, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_exact_columns(frame, PROBE_PREDICTION_COLUMNS, f"probe {path}")
    required = {
        "patient_id",
        "seed_base",
        "fold",
        "effective_seed",
        "split",
        "model",
        "task",
        "timepoint",
        "transition",
        "representation",
        "input_variant",
        "target",
        "feature_dim",
        "y_true",
        "y_pred",
        "y_true_standardized",
        "y_pred_standardized",
        "b0_prediction",
        "b0_prediction_standardized",
        "selected_alpha",
        "target_transform",
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "static_transform_sha256",
        "change_transform_sha256",
        "raw_target_file_sha256",
        "raw_targets_sha256",
        "test_prediction_guard_enforced",
        "test_predict_call_count",
        *PROBE_FALSE_FLAGS,
    }
    _require_columns(frame, required, f"probe {path}")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str)
    output["model"] = output["model"].astype(str).str.upper()
    output["task"] = output["task"].astype(str).str.lower()
    output["target"] = output["target"].astype(str).str.lower()
    output["timepoint"] = (
        output["timepoint"]
        .where(output["timepoint"].notna(), "")
        .astype(str)
        .str.upper()
    )
    output["transition"] = (
        output["transition"]
        .where(output["transition"].notna(), "")
        .map(lambda value: _normalise_transition(value) if str(value).strip() else "")
    )
    output["fold"] = _integer_values(output["fold"], f"probe {path}.fold")
    output["seed_base"] = _integer_values(
        output["seed_base"], f"probe {path}.seed_base"
    )
    output["effective_seed"] = _integer_values(
        output["effective_seed"], f"probe {path}.effective_seed"
    )
    if (
        set(output["model"]) != {model}
        or set(output["fold"]) != {fold}
        or set(output["seed_base"]) != {seed}
        or set(output["effective_seed"]) != {seed + fold}
    ):
        raise AnalysisInputError(f"probe path/model/seed/fold contract 不一致: {path}")
    if set(output["split"].astype(str).str.lower()) != {"test"}:
        raise AnalysisInputError(f"probe 只能含 outer-test rows: {path}")
    if set(output["target"]) != {"ftv"}:
        raise AnalysisInputError(f"正式 probe 必须 FTV-only: {path}")
    if set(output["representation"].astype(str).str.lower()) != {"response_state"}:
        raise AnalysisInputError(f"probe representation 必须是 response_state: {path}")
    if set(_integer_values(output["feature_dim"], f"probe {path}.feature_dim")) != {
        192
    }:
        raise AnalysisInputError(f"probe feature_dim 必须是 192: {path}")
    if set(output["task"]) != {"static", "change"}:
        raise AnalysisInputError(f"probe task coverage 错误: {path}")
    static = output["task"].eq("static")
    if set(output.loc[static, "input_variant"].astype(str).str.lower()) != {
        "current"
    } or set(output.loc[~static, "input_variant"].astype(str).str.lower()) != {
        "observed_difference"
    }:
        raise AnalysisInputError(f"probe input_variant contract 漂移: {path}")
    output["cell"] = np.where(static, output["timepoint"], output["transition"])
    expected_cells = {("static", item) for item in TIMEPOINTS} | {
        ("change", item) for item in TRANSITIONS
    }
    observed_cells = set(output[["task", "cell"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise AnalysisInputError(
            f"probe cell coverage 错误 {path}: 缺={sorted(expected_cells-observed_cells)} "
            f"多={sorted(observed_cells-expected_cells)}"
        )
    key = ["patient_id", "fold", "model", "task", "cell", "target"]
    if output.duplicated(key).any():
        raise AnalysisInputError(f"probe patient/cell key 重复: {path}")
    cell_sets = [
        set(part["patient_id"]) for _, part in output.groupby(["task", "cell"])
    ]
    if not cell_sets or any(values != cell_sets[0] for values in cell_sets[1:]):
        raise AnalysisInputError(
            f"probe 同一 fold 的七个 cell patient 集不一致: {path}"
        )
    _finite(
        output,
        (
            "y_true",
            "y_pred",
            "y_true_standardized",
            "y_pred_standardized",
            "b0_prediction",
            "b0_prediction_standardized",
            "selected_alpha",
        ),
        f"probe {path}",
    )
    selected_alpha = _numeric(output["selected_alpha"], f"probe {path}.selected_alpha")
    if not np.isin(selected_alpha, np.asarray(PROBE_ALPHAS, dtype=float)).all():
        raise AnalysisInputError(f"probe selected_alpha 不在冻结 grid: {path}")
    expected_transform = np.where(
        static,
        "static:log_epsilon+winsor01_99+median_iqr:" + output["timepoint"].astype(str),
        "delta:log_epsilon_difference+winsor01_99+median_iqr",
    )
    if not np.array_equal(
        output["target_transform"].astype(str).to_numpy(), expected_transform
    ):
        raise AnalysisInputError(f"probe FTV target transform contract 漂移: {path}")
    for _, part in output.groupby(["task", "cell"], sort=False):
        if (
            part["selected_alpha"].nunique(dropna=False) != 1
            or part["target_transform"].nunique(dropna=False) != 1
            or part["b0_prediction"].nunique(dropna=False) != 1
            or part["b0_prediction_standardized"].nunique(dropna=False) != 1
        ):
            raise AnalysisInputError(
                f"probe 同一 cell 的 selection/transform/B0 非唯一: {path}"
            )
    for column in PROBE_FALSE_FLAGS:
        if _strict_bool(output[column], f"probe.{column}").any():
            raise AnalysisInputError(f"probe 声称 test 参与选择/拟合: {path}/{column}")
    if not _strict_bool(
        output["test_prediction_guard_enforced"], "probe.test_prediction_guard_enforced"
    ).all():
        raise AnalysisInputError(f"probe test-once guard 未执行: {path}")
    if (
        not pd.to_numeric(output["test_predict_call_count"], errors="coerce")
        .eq(1)
        .all()
    ):
        raise AnalysisInputError(f"probe test predict call count 非 1: {path}")
    for column in (
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "static_transform_sha256",
        "change_transform_sha256",
        "raw_target_file_sha256",
        "raw_targets_sha256",
    ):
        _require_sha(output[column], f"probe.{column}")
    if set(output["fold_manifest_sha256"].astype(str).str.lower()) != {
        EXPECTED_FOLD_MANIFEST_SHA256
    }:
        raise AnalysisInputError(f"probe fold manifest SHA 漂移: {path}")
    return output


def _normalise_pcr_file(path: Path, seed: int, model: str, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_exact_columns(frame, PCR_PREDICTION_COLUMNS, f"pCR {path}")
    required = {
        "patient_id",
        "seed_base",
        "fold",
        "effective_seed",
        "split",
        "model",
        "decision_point",
        "y_true",
        "probability",
        "predicted_label",
        "threshold",
        "penalty",
        "C",
        "readout",
        "class_weight",
        "feature_schema",
        "feature_schema_sha256",
        "feature_dim",
        "val_auroc",
        "val_auprc",
        "val_youden",
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "test_feature_matrix_constructed_after_selection_lock",
        "test_prediction_guard_enforced",
        "test_predict_proba_call_count",
        *PCR_FALSE_FLAGS,
    }
    _require_columns(frame, required, f"pCR {path}")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str)
    output["model"] = output["model"].astype(str).str.upper()
    output["fold"] = _integer_values(output["fold"], f"pCR {path}.fold")
    output["decision_point"] = output["decision_point"].map(_normalise_decision)
    output["seed_base"] = _integer_values(output["seed_base"], f"pCR {path}.seed_base")
    output["effective_seed"] = _integer_values(
        output["effective_seed"], f"pCR {path}.effective_seed"
    )
    if (
        set(output["model"]) != {model}
        or set(output["fold"]) != {fold}
        or set(output["seed_base"]) != {seed}
        or set(output["effective_seed"]) != {seed + fold}
    ):
        raise AnalysisInputError(f"pCR path/model/seed/fold contract 不一致: {path}")
    if set(output["split"].astype(str).str.lower()) != {"test"}:
        raise AnalysisInputError(f"pCR 只能含 outer-test rows: {path}")
    if set(output["decision_point"]) != set(DECISION_POINTS):
        raise AnalysisInputError(f"pCR decision point coverage 错误: {path}")
    if set(output["readout"].astype(str).str.lower()) != {
        "class-balanced logisticregression"
    }:
        raise AnalysisInputError(f"pCR readout 算法 contract 漂移: {path}")
    if set(output["class_weight"].astype(str).str.lower()) != {"balanced"}:
        raise AnalysisInputError(f"pCR class_weight contract 漂移: {path}")
    for point, expected in PCR_FEATURE_SCHEMAS.items():
        observed = set(
            output.loc[output["decision_point"].eq(point), "feature_schema"].astype(str)
        )
        if observed != {expected}:
            raise AnalysisInputError(
                f"pCR feature schema 错误: {path}/{point}: {observed}"
            )
        expected_dim = PCR_FEATURE_DIMS[point]
        if set(
            _integer_values(
                output.loc[output["decision_point"].eq(point), "feature_dim"],
                f"pCR {path}.{point}.feature_dim",
            )
        ) != {expected_dim}:
            raise AnalysisInputError(f"pCR feature dimension 错误: {path}/{point}")
        expected_schema_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        if set(
            output.loc[output["decision_point"].eq(point), "feature_schema_sha256"]
            .astype(str)
            .str.lower()
        ) != {expected_schema_sha}:
            raise AnalysisInputError(f"pCR feature schema SHA 错误: {path}/{point}")
    key = ["patient_id", "fold", "model", "decision_point"]
    if output.duplicated(key).any():
        raise AnalysisInputError(f"pCR patient/decision key 重复: {path}")
    point_sets = [
        set(part["patient_id"]) for _, part in output.groupby("decision_point")
    ]
    if not point_sets or any(values != point_sets[0] for values in point_sets[1:]):
        raise AnalysisInputError(f"pCR 三个 decision point patient 集不一致: {path}")
    _finite(
        output,
        (
            "y_true",
            "probability",
            "predicted_label",
            "threshold",
            "C",
            "val_auroc",
            "val_auprc",
            "val_youden",
        ),
        f"pCR {path}",
    )
    probability = _numeric(output["probability"], f"pCR {path}.probability")
    threshold = _numeric(output["threshold"], f"pCR {path}.threshold")
    if not ((probability >= 0) & (probability <= 1)).all():
        raise AnalysisInputError(f"pCR probability 超出 [0,1]: {path}")
    if not ((threshold >= 0) & (threshold <= 1)).all():
        raise AnalysisInputError(f"pCR threshold 超出 [0,1]: {path}")
    truth = _integer_values(output["y_true"], f"pCR {path}.y_true")
    label = _integer_values(output["predicted_label"], f"pCR {path}.predicted_label")
    if not set(truth).issubset({0, 1}):
        raise AnalysisInputError(f"pCR y_true 非 0/1: {path}")
    output["y_true"] = truth
    if output.groupby("patient_id")["y_true"].nunique(dropna=False).ne(1).any():
        raise AnalysisInputError(
            f"pCR 同一 patient 跨 decision point y_true 漂移: {path}"
        )
    if not set(label).issubset({0, 1}):
        raise AnalysisInputError(f"pCR predicted_label 非 0/1: {path}")
    if not np.array_equal(label, (probability >= threshold).astype(np.int64)):
        raise AnalysisInputError(
            f"pCR predicted_label 与 probability/threshold 不一致: {path}"
        )
    output["probability"] = probability
    output["threshold"] = threshold
    output["predicted_label"] = label
    output["penalty"] = output["penalty"].astype(str).str.lower()
    if not set(output["penalty"]).issubset(set(PCR_PENALTIES)):
        raise AnalysisInputError(f"pCR penalty 不在冻结 grid: {path}")
    c_values = _numeric(output["C"], f"pCR {path}.C")
    if not np.isin(c_values, np.asarray(PCR_C_GRID, dtype=float)).all():
        raise AnalysisInputError(f"pCR C 不在冻结 grid: {path}")
    output["C"] = c_values
    val_auroc = _numeric(output["val_auroc"], f"pCR {path}.val_auroc")
    val_auprc = _numeric(output["val_auprc"], f"pCR {path}.val_auprc")
    val_youden = _numeric(output["val_youden"], f"pCR {path}.val_youden")
    if (
        not ((val_auroc >= 0) & (val_auroc <= 1)).all()
        or not ((val_auprc >= 0) & (val_auprc <= 1)).all()
        or not ((val_youden >= -1) & (val_youden <= 1)).all()
    ):
        raise AnalysisInputError(f"pCR validation metric 超出合法范围: {path}")
    for _, part in output.groupby("decision_point", sort=False):
        frozen_columns = (
            "threshold",
            "penalty",
            "C",
            "val_auroc",
            "val_auprc",
            "val_youden",
            "feature_schema_sha256",
        )
        if any(part[column].nunique(dropna=False) != 1 for column in frozen_columns):
            raise AnalysisInputError(
                f"pCR 同一 decision point selection 值非唯一: {path}"
            )
    for column in PCR_FALSE_FLAGS:
        if _strict_bool(output[column], f"pCR.{column}").any():
            raise AnalysisInputError(f"pCR 声称 test 参与选择/拟合: {path}/{column}")
    for column in (
        "test_feature_matrix_constructed_after_selection_lock",
        "test_prediction_guard_enforced",
    ):
        if not _strict_bool(output[column], f"pCR.{column}").all():
            raise AnalysisInputError(
                f"pCR test-once/selection-lock guard 未执行: {path}/{column}"
            )
    if (
        not pd.to_numeric(output["test_predict_proba_call_count"], errors="coerce")
        .eq(1)
        .all()
    ):
        raise AnalysisInputError(f"pCR test predict_proba call count 非 1: {path}")
    for column in (
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "feature_schema_sha256",
    ):
        _require_sha(output[column], f"pCR.{column}")
    if set(output["fold_manifest_sha256"].astype(str).str.lower()) != {
        EXPECTED_FOLD_MANIFEST_SHA256
    }:
        raise AnalysisInputError(f"pCR fold manifest SHA 漂移: {path}")
    return output


def _validate_prediction_closure(probe: pd.DataFrame, pcr: pd.DataFrame) -> None:
    probe_key = ["patient_id", "fold", "task", "cell"]
    pcr_key = ["patient_id", "fold", "decision_point"]
    for seed in SEEDS:
        for frame, key, label, expected_n in (
            (probe, probe_key, "probe", 375),
            (pcr, pcr_key, "pCR", 808),
        ):
            seed_part = frame.loc[frame["seed_base"].eq(seed)]
            for model in MODELS:
                model_part = seed_part.loc[seed_part["model"].eq(model)]
                if model_part["patient_id"].nunique() != expected_n:
                    raise AnalysisInputError(
                        f"{label} seed={seed}/{model} unique patient != {expected_n}"
                    )
                # 每位患者在固定 outer folds 中恰好一次作为 test。
                assignments = model_part[["patient_id", "fold"]].drop_duplicates()
                if assignments["patient_id"].duplicated().any():
                    raise AnalysisInputError(
                        f"{label} patient 跨 fold 重复: seed={seed}/{model}"
                    )
            left = (
                seed_part.loc[seed_part["model"].eq("G1")]
                .sort_values(key)
                .reset_index(drop=True)
            )
            right = (
                seed_part.loc[seed_part["model"].eq("G3")]
                .sort_values(key)
                .reset_index(drop=True)
            )
            if not left[key].equals(right[key]):
                raise AnalysisInputError(
                    f"{label} G1/G3 exact-patient closure 失败: seed={seed}"
                )
            truth_columns = (
                ("y_true", "y_true_standardized") if label == "probe" else ("y_true",)
            )
            if any(
                not np.array_equal(left[column].to_numpy(), right[column].to_numpy())
                for column in truth_columns
            ):
                raise AnalysisInputError(f"{label} G1/G3 target 不一致: seed={seed}")
            if label == "probe" and any(
                not np.allclose(left[column], right[column], rtol=0.0, atol=1e-12)
                for column in ("b0_prediction", "b0_prediction_standardized")
            ):
                raise AnalysisInputError(f"probe G1/G3 B0 不一致: seed={seed}")
    # fixed folds/targets 必须跨 representation training seeds 不变。
    for frame, keys, sort_keys, label in (
        (
            probe,
            [
                "patient_id",
                "fold",
                "task",
                "cell",
                "y_true",
                "y_true_standardized",
                "b0_prediction",
                "b0_prediction_standardized",
            ],
            ["patient_id", "fold", "task", "cell"],
            "probe",
        ),
        (
            pcr,
            ["patient_id", "fold", "decision_point", "y_true"],
            ["patient_id", "fold", "decision_point"],
            "pCR",
        ),
    ):
        reference = (
            frame.loc[frame["seed_base"].eq(SEEDS[0]) & frame["model"].eq("G1"), keys]
            .sort_values(sort_keys)
            .reset_index(drop=True)
        )
        for seed in SEEDS[1:]:
            current = (
                frame.loc[frame["seed_base"].eq(seed) & frame["model"].eq("G1"), keys]
                .sort_values(sort_keys)
                .reset_index(drop=True)
            )
            if not reference.equals(current):
                raise AnalysisInputError(
                    f"{label} patient/fold/target 跨 seed 漂移: seed={seed}"
                )


def discover_predictions(
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe_parts: list[pd.DataFrame] = []
    pcr_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for seed, model, fold in sorted(_expected_grid()):
        for kind, normaliser, target in (
            ("representation_probes", _normalise_probe_file, probe_parts),
            ("pcr_readouts", _normalise_pcr_file, pcr_parts),
        ):
            path = _prediction_path(config.prediction_root, kind, seed, model, fold)
            if not path.is_file():
                raise AnalysisInputError(f"缺正式 prediction: {path}")
            frame = normaliser(path, seed, model, fold)
            target.append(frame)
            manifest_rows.append(
                _manifest_row(
                    (
                        "probe_prediction"
                        if kind == "representation_probes"
                        else "pcr_prediction"
                    ),
                    path,
                    seed_base=seed,
                    model=model,
                    fold=fold,
                    rows=len(frame),
                    patients=frame["patient_id"].nunique(),
                )
            )
    probe = pd.concat(probe_parts, ignore_index=True)
    pcr = pd.concat(pcr_parts, ignore_index=True)
    if len(probe) != EXPECTED_PROBE_ROWS:
        raise AnalysisInputError(f"probe 总行数 {len(probe)} != {EXPECTED_PROBE_ROWS}")
    if len(pcr) != EXPECTED_PCR_ROWS:
        raise AnalysisInputError(f"pCR 总行数 {len(pcr)} != {EXPECTED_PCR_ROWS}")
    _validate_prediction_closure(probe, pcr)
    return probe, pcr, pd.DataFrame(manifest_rows)


def _all_tensors_finite(value: Any) -> tuple[bool, int]:
    """递归检查 checkpoint 内全部 tensor；返回 finite 与 tensor 数。"""

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()), 1
    if isinstance(value, Mapping):
        states = [_all_tensors_finite(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        states = [_all_tensors_finite(item) for item in value]
    else:
        return True, 0
    return all(state for state, _ in states), sum(count for _, count in states)


def _hash_ordered_rows(*arrays: Sequence[Any]) -> str:
    lines = zip(*(map(str, values) for values in arrays), strict=True)
    return hashlib.sha256(
        "\n".join("\t".join(row) for row in lines).encode()
    ).hexdigest()


def _compare_common_checkpoint_contract(
    g1: Mapping[str, Any], g3: Mapping[str, Any], label: str
) -> None:
    for name in ("implementation_sha256", "source_commit"):
        if str(g1.get(name)) != str(g3.get(name)):
            raise AnalysisInputError(f"{label} G1/G3 {name} 不一致")
    if str(g1.get("shared_initialization_sha256")) != str(
        g3.get("shared_initialization_sha256")
    ):
        raise AnalysisInputError(f"{label} G1/G3 shared initialization SHA 不一致")
    if dict(g1.get("split_hashes", {})) != dict(g3.get("split_hashes", {})):
        raise AnalysisInputError(f"{label} G1/G3 split hashes 不一致")
    if str(g1.get("ftv_transform_sha256")) != str(g3.get("ftv_transform_sha256")):
        raise AnalysisInputError(f"{label} G1/G3 FTV transform SHA 不一致")
    if dict(g1.get("data_contract", {})) != dict(g3.get("data_contract", {})):
        raise AnalysisInputError(f"{label} G1/G3 data_contract 不一致")
    model1, model3 = dict(g1.get("model_config", {})), dict(g3.get("model_config", {}))
    for name in ("model_name", "direct_ftv_grounding"):
        model1.pop(name, None)
        model3.pop(name, None)
    if model1 != model3:
        raise AnalysisInputError(f"{label} G1/G3 common model config 不一致")
    if dict(g1.get("train_config", {})) != dict(g3.get("train_config", {})):
        raise AnalysisInputError(f"{label} G1/G3 train config 不一致")
    loss1, loss3 = dict(g1.get("loss_config", {})), dict(g3.get("loss_config", {}))
    loss1.pop("lambda_ftv", None)
    loss3.pop("lambda_ftv", None)
    if loss1 != loss3:
        raise AnalysisInputError(f"{label} G1/G3 common loss config 不一致")
    if dict(g1.get("runtime", {})) != dict(g3.get("runtime", {})):
        raise AnalysisInputError(f"{label} G1/G3 patient-order/runtime policy 不一致")
    if dict(g1.get("determinism", {})) != dict(g3.get("determinism", {})):
        raise AnalysisInputError(f"{label} G1/G3 determinism/generator policy 不一致")


def _require_live_declared_file(path_value: Any, sha_value: Any, label: str) -> None:
    path = Path(str(path_value)).expanduser()
    if not path.is_file() or file_sha256(path) != str(sha_value):
        raise AnalysisInputError(f"{label} 文件缺失或 live SHA 不一致: {path}")


def _load_probe_target_contract(
    selection: pd.DataFrame,
    fold: int,
    feature: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    """从冻结 raw target/transform 独立重建 probe target contract。"""

    raw_path = (
        REPO_ROOT
        / "additional_experiments"
        / "radiomics_next_change"
        / "data_audit"
        / "radiomics_transition_targets_raw.csv"
    ).resolve()
    static_path = (
        REPO_ROOT
        / "additional_experiments"
        / "observed_state_radiomics_audit"
        / "configs"
        / f"static_target_transform_fold_{fold}.json"
    ).resolve()
    change_path = (
        REPO_ROOT
        / "additional_experiments"
        / "radiomics_next_change"
        / "configs"
        / f"radiomics_transform_fold_{fold}.json"
    ).resolve()
    expected_paths = {
        "raw_target_file": (raw_path, EXPECTED_RAW_TARGET_FILE_SHA256),
        "static_transform_file": (
            static_path,
            EXPECTED_STATIC_TRANSFORM_SHA256[fold],
        ),
        "change_transform_file": (
            change_path,
            EXPECTED_CHANGE_TRANSFORM_SHA256[fold],
        ),
    }
    for column, (expected_path, expected_sha) in expected_paths.items():
        observed = (
            selection[column].astype(str).map(lambda value: Path(value).resolve())
        )
        if (
            set(observed) != {expected_path}
            or file_sha256(expected_path) != expected_sha
        ):
            raise AnalysisInputError(
                f"probe 冻结 target asset 路径/SHA 漂移: fold={fold}/{column}"
            )
    if (
        set(selection["raw_target_file_sha256"].astype(str))
        != {EXPECTED_RAW_TARGET_FILE_SHA256}
        or set(selection["raw_targets_sha256"].astype(str))
        != {EXPECTED_PROBE_RAW_TARGETS_SHA256}
        or set(selection["static_transform_sha256"].astype(str))
        != {EXPECTED_STATIC_TRANSFORM_SHA256[fold]}
        or set(selection["change_transform_sha256"].astype(str))
        != {EXPECTED_CHANGE_TRANSFORM_SHA256[fold]}
    ):
        raise AnalysisInputError(f"probe 冻结 target SHA 声明漂移: fold={fold}")

    static_payload = _read_json(static_path, "static target transform")
    change_payload = _read_json(change_path, "change target transform")
    train_ids = feature["patient_ids"][feature["splits"] == "train"].astype(str)
    train_hash = patient_hash(train_ids)
    feature_order = ("ftv", "sphericity", "ld", "bpe")
    if (
        int(static_payload.get("schema_version", -1)) != 1
        or int(static_payload.get("fold", -1)) != fold
        or tuple(static_payload.get("feature_order", ())) != feature_order
        or tuple(static_payload.get("timepoints", ())) != TIMEPOINTS
        or static_payload.get("raw_targets_sha256") != EXPECTED_PROBE_RAW_TARGETS_SHA256
        or static_payload.get("train_patient_hash") != train_hash
        or change_payload.get("spec_version")
        != "adjacent_v2_ftv_ld_logepsilon_sphericity_bpe_absolute_winsor01_99_robust"
        or int(change_payload.get("fold", -1)) != fold
        or tuple(change_payload.get("feature_order", ())) != feature_order
        or change_payload.get("raw_targets_sha256") != EXPECTED_PROBE_RAW_TARGETS_SHA256
        or change_payload.get("train_patient_hash") != train_hash
    ):
        raise AnalysisInputError(
            f"probe target transform train-only contract 漂移: fold={fold}"
        )
    try:
        static_specs = {
            (str(row["timepoint"]), str(row["feature_name"])): row
            for row in static_payload["specs"]
        }
        change_specs = {str(row["name"]): row for row in change_payload["features"]}
    except (KeyError, TypeError) as exc:
        raise AnalysisInputError(
            f"probe target transform schema 非法: fold={fold}"
        ) from exc
    if set(static_specs) != {
        (timepoint, feature_name)
        for timepoint in TIMEPOINTS
        for feature_name in feature_order
    } or set(change_specs) != set(feature_order):
        raise AnalysisInputError(
            f"probe target transform cell coverage 漂移: fold={fold}"
        )
    for spec in [*static_specs.values(), *change_specs.values()]:
        feature_name = str(spec.get("feature_name", spec.get("name", "")))
        expected_value_transform = (
            "log_epsilon" if feature_name in {"ftv", "ld"} else "identity"
        )
        try:
            numeric = np.asarray(
                [
                    float(spec["epsilon"]),
                    float(spec["winsor_low"]),
                    float(spec["winsor_high"]),
                    float(spec["center"]),
                    float(spec["scale"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisInputError(
                f"probe target transform numeric schema 非法: fold={fold}"
            ) from exc
        if (
            spec.get("value_transform") != expected_value_transform
            or not np.isfinite(numeric).all()
            or (
                numeric[0] <= 0
                if expected_value_transform == "log_epsilon"
                else numeric[0] != 0
            )
            or numeric[2] < numeric[1]
            or numeric[4] <= 0
        ):
            raise AnalysisInputError(f"probe target transform 参数非法: fold={fold}")

    raw = pd.read_csv(raw_path, dtype={"patient_id": str}, low_memory=False)
    required = {
        "patient_id",
        "transition",
        "start_visit",
        "end_visit",
        "ftv_start",
        "ftv_end",
        "ftv_valid",
    }
    _require_columns(raw, required, f"probe raw target {raw_path}")
    raw["transition"] = raw["transition"].map(_normalise_transition)
    raw["ftv_valid"] = _strict_bool(raw["ftv_valid"], "probe raw target.ftv_valid")
    if (
        len(raw) != 1125
        or raw["patient_id"].nunique() != 375
        or raw.duplicated(["patient_id", "transition"]).any()
    ):
        raise AnalysisInputError("probe raw target patient/transition coverage 漂移")
    raw_targets: dict[str, np.ndarray] = {}
    for patient_id, group in raw.groupby("patient_id", sort=False):
        indexed = group.set_index("transition").reindex(TRANSITIONS)
        if indexed[["start_visit", "end_visit"]].isna().any().any():
            raise AnalysisInputError(f"probe raw target transition 缺失: {patient_id}")
        values = np.empty((3, 3), dtype=np.float64)
        for index, transition in enumerate(TRANSITIONS):
            row = indexed.loc[transition]
            start, end = transition.split("→")
            if str(row["start_visit"]) != start or str(row["end_visit"]) != end:
                raise AnalysisInputError(
                    f"probe raw target transition endpoint 错位: {patient_id}/{transition}"
                )
            try:
                values[index] = (
                    float(row["ftv_start"]),
                    float(row["ftv_end"]),
                    float(bool(row["ftv_valid"])),
                )
            except (TypeError, ValueError) as exc:
                raise AnalysisInputError(
                    f"probe raw target FTV 非数值: {patient_id}/{transition}"
                ) from exc
        for index in (0, 1):
            both = bool(values[index, 2]) and bool(values[index + 1, 2])
            if both and not math.isclose(
                float(values[index, 1]),
                float(values[index + 1, 0]),
                rel_tol=0,
                abs_tol=1e-10,
            ):
                raise AnalysisInputError(
                    f"probe raw target shared visit 漂移: {patient_id}"
                )
        raw_targets[str(patient_id)] = values
    return raw_targets, static_specs, change_specs


def _probe_target_value(
    patient_id: str,
    task: str,
    cell: str,
    raw_targets: Mapping[str, np.ndarray],
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, bool]:
    values = raw_targets.get(str(patient_id))
    if values is None:
        return math.nan, math.nan, False
    if task == "static":
        index = TIMEPOINTS.index(cell)
        natural = (values[0, 0], values[0, 1], values[1, 1], values[2, 1])[index]
        valid = (
            bool(values[0, 2]),
            bool(values[0, 2]) and bool(values[1, 2]),
            bool(values[1, 2]) and bool(values[2, 2]),
            bool(values[2, 2]),
        )[index]
        spec = static_specs[(cell, "ftv")]
        if not valid or not math.isfinite(float(natural)):
            return math.nan, math.nan, False
        analysis_value = math.log(float(natural) + float(spec["epsilon"]))
    else:
        index = TRANSITIONS.index(cell)
        start, end, valid_value = values[index]
        spec = change_specs["ftv"]
        valid = bool(valid_value) and np.isfinite((start, end)).all()
        if not valid:
            return math.nan, math.nan, False
        epsilon = float(spec["epsilon"])
        if start + epsilon <= 0 or end + epsilon <= 0:
            raise AnalysisInputError(
                f"probe raw target log domain 非正: {patient_id}/{cell}"
            )
        analysis_value = math.log(float(end) + epsilon) - math.log(
            float(start) + epsilon
        )
        natural = analysis_value
    clipped = float(
        np.clip(
            analysis_value,
            float(spec["winsor_low"]),
            float(spec["winsor_high"]),
        )
    )
    standardized = (clipped - float(spec["center"])) / float(spec["scale"])
    return float(standardized), float(natural), True


def _probe_inverse_prediction(
    standardized: np.ndarray,
    task: str,
    cell: str,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    spec = static_specs[(cell, "ftv")] if task == "static" else change_specs["ftv"]
    analysis_value = np.asarray(standardized, dtype=np.float64) * float(
        spec["scale"]
    ) + float(spec["center"])
    if task == "static":
        return np.exp(analysis_value) - float(spec["epsilon"])
    return analysis_value


def _probe_cell_matrix(
    response: np.ndarray, indices: np.ndarray, task: str, cell: str
) -> np.ndarray:
    index = TIMEPOINTS.index(cell) if task == "static" else TRANSITIONS.index(cell)
    if task == "static":
        matrix = response[indices, index]
    else:
        matrix = response[indices, index + 1] - response[indices, index]
    return np.asarray(matrix, dtype=np.float64)


def _pcr_readout_matrix(response: np.ndarray, decision_point: str) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 3 or response.shape[1:] != (4, 192):
        raise AnalysisInputError(
            f"pCR independent response shape 非法: {response.shape}"
        )
    if decision_point == "T0":
        matrix = response[:, 0]
    elif decision_point == "T0-T1":
        r0, r1 = response[:, 0], response[:, 1]
        matrix = np.concatenate((r0, r1, r1 - r0), axis=1)
    elif decision_point == "T0-T2":
        r0, r1, r2 = response[:, 0], response[:, 1], response[:, 2]
        matrix = np.concatenate((r0, r1, r2, r1 - r0, r2 - r1, r2 - r0), axis=1)
    else:
        raise AnalysisInputError(
            f"pCR independent decision point 非法: {decision_point}"
        )
    if (
        matrix.shape[1] != PCR_FEATURE_DIMS[decision_point]
        or not np.isfinite(matrix).all()
    ):
        raise AnalysisInputError(
            f"pCR independent matrix contract 失败: {decision_point}"
        )
    return matrix


def _independent_youden(
    labels: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        raise AnalysisInputError("pCR validation Youden 缺一个 class")
    rows: list[tuple[float, float, float, float]] = []
    for threshold in np.unique(np.concatenate(([0.0, 1.0], probability))):
        predicted = probability >= threshold
        sensitivity = float(np.count_nonzero(predicted & (labels == 1)) / positives)
        specificity = float(np.count_nonzero(~predicted & (labels == 0)) / negatives)
        rows.append(
            (
                float(threshold),
                sensitivity + specificity - 1.0,
                sensitivity,
                specificity,
            )
        )
    best = max(row[1] for row in rows)
    return min(
        (row for row in rows if row[1] >= best - 1e-12),
        key=lambda row: (abs(row[0] - 0.5), row[0]),
    )


def _validate_probe_selection_protocol(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    predictions: pd.DataFrame,
    selection_path: Path,
    prediction_path: Path,
    expected: Mapping[str, str],
    feature: Mapping[str, np.ndarray],
) -> None:
    label = f"probe selection {selection_path}"
    _require_exact_columns(frame, PROBE_SELECTION_COLUMNS, label)
    if summary.get("prediction_columns") != list(
        PROBE_PREDICTION_COLUMNS
    ) or summary.get("selection_columns") != list(PROBE_SELECTION_COLUMNS):
        raise AnalysisInputError(f"probe summary column schema 漂移: {selection_path}")
    if (
        summary.get("targets") != ["ftv"]
        or summary.get("tasks") != ["static", "change"]
        or summary.get("representation") != "response_state"
        or summary.get("static_feature") != "r_t"
        or summary.get("delta_feature") != "r_(t+1)-r_t"
        or int(summary.get("probe_cells", -1)) != 7
        or int(summary.get("test_prediction_rows", -1)) != len(predictions)
        or int(summary.get("test_patient_count", -1))
        != predictions["patient_id"].nunique()
        or str(summary.get("probe_implementation_sha256"))
        != probe_implementation_sha256()
        or str(summary.get("sklearn_version")) != sklearn.__version__
    ):
        raise AnalysisInputError(
            f"probe summary protocol/count/implementation 漂移: {selection_path}"
        )
    expected_ridge = {
        "type": "single_output",
        "alphas": [float(value) for value in PROBE_ALPHAS],
        "solver": RIDGE_SOLVER,
        "tol": RIDGE_TOL,
        "max_iter": RIDGE_MAX_ITER,
        "feature_scaler": "StandardScaler fit on fold train only",
        "alpha_selection": (
            "validation standardized MSE; <=1e-12 tie chooses smaller alpha"
        ),
    }
    if dict(summary.get("ridge", {})) != expected_ridge:
        raise AnalysisInputError(
            f"probe summary Ridge/grid protocol 漂移: {selection_path}"
        )
    expected_guards = {
        "feature_scaler_fit_scope": "outer fold train only",
        "ridge_fit_scope": "outer fold train only",
        "alpha_selection_scope": "outer fold validation only",
        "test_constructed_after_alpha_lock": True,
        "test_predict_calls_per_cell": 1,
        "test_prediction_guard_enforced": True,
        "test_used_for_target_transform": False,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_lambda_selection": False,
        "test_used_for_scaler": False,
        "test_used_for_alpha_selection": False,
        "radiomics_used_as_input": False,
        "ftv_head_output_used_as_input": False,
        "transition_prediction_used_as_input": False,
        "world_model_trained_or_finetuned": False,
    }
    if dict(summary.get("leakage_guards", {})) != expected_guards:
        raise AnalysisInputError(f"probe summary leakage guards 漂移: {selection_path}")

    current = frame.copy()
    current["task"] = current["task"].astype(str).str.lower()
    current["timepoint"] = (
        current["timepoint"]
        .where(current["timepoint"].notna(), "")
        .astype(str)
        .str.upper()
    )
    current["transition"] = (
        current["transition"]
        .where(current["transition"].notna(), "")
        .map(lambda value: _normalise_transition(value) if str(value).strip() else "")
    )
    current["cell"] = np.where(
        current["task"].eq("static"), current["timepoint"], current["transition"]
    )
    expected_cells = {("static", item) for item in TIMEPOINTS} | {
        ("change", item) for item in TRANSITIONS
    }
    if (
        len(current) != 7
        or current.duplicated(["task", "cell", "target"]).any()
        or set(current[["task", "cell"]].itertuples(index=False, name=None))
        != expected_cells
        or set(current["target"].astype(str).str.lower()) != {"ftv"}
        or set(current["representation"].astype(str)) != {"response_state"}
        or set(_integer_values(current["feature_dim"], f"{label}.feature_dim")) != {192}
    ):
        raise AnalysisInputError(
            f"probe selection 七个冻结 FTV cells/schema 错误: {selection_path}"
        )
    static = current["task"].eq("static")
    if set(current.loc[static, "input_variant"].astype(str)) != {"current"} or set(
        current.loc[~static, "input_variant"].astype(str)
    ) != {"observed_difference"}:
        raise AnalysisInputError(
            f"probe selection input_variant 漂移: {selection_path}"
        )
    expected_transform = np.where(
        static,
        "static:log_epsilon+winsor01_99+median_iqr:" + current["timepoint"].astype(str),
        "delta:log_epsilon_difference+winsor01_99+median_iqr",
    )
    if not np.array_equal(current["target_transform"].astype(str), expected_transform):
        raise AnalysisInputError(
            f"probe selection target transform 漂移: {selection_path}"
        )
    selected_alphas = _numeric(current["selected_alpha"], f"{label}.selected_alpha")
    if not np.isin(selected_alphas, np.asarray(PROBE_ALPHAS, dtype=float)).all():
        raise AnalysisInputError(
            f"probe selection alpha 不在冻结 grid: {selection_path}"
        )
    _finite(
        current,
        (
            "val_mse_standardized",
            "ridge_tol",
            "ridge_max_iter",
            "ridge_intercept",
            "train_target_mean_standardized",
            "b0_val_mse_standardized",
        ),
        label,
    )
    if (_numeric(current["val_mse_standardized"], f"{label}.val_mse") < 0).any() or (
        _numeric(current["b0_val_mse_standardized"], f"{label}.b0_val_mse") < 0
    ).any():
        raise AnalysisInputError(
            f"probe validation MSE 非负 contract 失败: {selection_path}"
        )
    if (
        set(current["ridge_solver"].astype(str)) != {RIDGE_SOLVER}
        or not np.equal(
            _numeric(current["ridge_tol"], f"{label}.ridge_tol"), RIDGE_TOL
        ).all()
        or not np.equal(
            _integer_values(current["ridge_max_iter"], f"{label}.ridge_max_iter"),
            RIDGE_MAX_ITER,
        ).all()
        or not _strict_bool(
            current["ridge_fit_intercept"], f"{label}.fit_intercept"
        ).all()
        or set(current["sklearn_version"].astype(str)) != {sklearn.__version__}
    ):
        raise AnalysisInputError(
            f"probe Ridge solver/tol/max_iter/intercept 漂移: {selection_path}"
        )
    counts = {
        name: _integer_values(current[name], f"{label}.{name}")
        for name in ("n_train", "n_val", "n_test", "feature_scaler_n_samples_seen")
    }
    if (
        any((values <= 0).any() for values in counts.values())
        or not np.array_equal(
            counts["n_train"], counts["feature_scaler_n_samples_seen"]
        )
        or not np.equal(counts["n_test"], predictions["patient_id"].nunique()).all()
    ):
        raise AnalysisInputError(
            f"probe train/val/test/scaler counts 不闭合: {selection_path}"
        )
    alpha_keys = {format(value, ".17g") for value in PROBE_ALPHAS}
    for row in current.itertuples(index=False):
        grid = _json_mapping(row.alpha_validation_mse_json, f"{label}.alpha_grid")
        try:
            scores = {float(key): float(value) for key, value in grid.items()}
        except (TypeError, ValueError) as exc:
            raise AnalysisInputError(f"{label}.alpha_grid 含非数值") from exc
        if (
            set(grid) != alpha_keys
            or not np.isfinite(list(scores.values())).all()
            or any(score < 0 for score in scores.values())
        ):
            raise AnalysisInputError(
                f"probe alpha grid evidence 不完整: {selection_path}"
            )
        best = min(scores.values())
        selected = min(
            alpha for alpha, score in scores.items() if score <= best + 1e-12
        )
        if not math.isclose(
            float(row.selected_alpha), selected, rel_tol=0, abs_tol=0
        ) or not math.isclose(
            float(row.val_mse_standardized), scores[selected], rel_tol=0, abs_tol=1e-12
        ):
            raise AnalysisInputError(
                f"probe validation-only alpha selection 不可复算: {selection_path}"
            )
        _json_finite_vector(row.ridge_coef_json, f"{label}.ridge_coef", 192)
        _json_finite_vector(row.feature_scaler_mean_json, f"{label}.scaler_mean", 192)
        scale = _json_finite_vector(
            row.feature_scaler_scale_json, f"{label}.scaler_scale", 192
        )
        if (scale <= 0).any():
            raise AnalysisInputError(f"probe scaler scale 非正: {selection_path}")
    for path_column, sha_column in (
        ("static_transform_file", "static_transform_sha256"),
        ("change_transform_file", "change_transform_sha256"),
        ("raw_target_file", "raw_target_file_sha256"),
    ):
        pairs = current[[path_column, sha_column]].drop_duplicates()
        if len(pairs) != 1:
            raise AnalysisInputError(
                f"probe {path_column}/{sha_column} 非唯一: {selection_path}"
            )
        _require_live_declared_file(
            pairs.iloc[0][path_column],
            pairs.iloc[0][sha_column],
            f"probe {path_column}",
        )
    for column in (
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "static_transform_sha256",
        "change_transform_sha256",
        "raw_target_file_sha256",
        "raw_targets_sha256",
    ):
        _require_sha(current[column], f"{label}.{column}")
    if (
        set(current["source_feature_sha256"].astype(str))
        != {expected["feature_sha256"]}
        or set(current["source_checkpoint_sha256"].astype(str))
        != {expected["checkpoint_sha256"]}
        or set(current["feature_extractor_sha256"].astype(str))
        != {expected["feature_extractor_sha256"]}
        or set(current["canonical_patient_order_sha256"].astype(str))
        != {expected["canonical_patient_order_sha256"]}
        or set(current["canonical_patient_label_sha256"].astype(str))
        != {expected["canonical_patient_label_sha256"]}
        or set(current["fold_manifest_sha256"].astype(str))
        != {EXPECTED_FOLD_MANIFEST_SHA256}
    ):
        raise AnalysisInputError(
            f"probe selection source/canonical hash 未闭合: {selection_path}"
        )
    for key, summary_key in (
        ("source_feature_sha256", "source_feature_sha256"),
        ("source_checkpoint_sha256", "source_checkpoint_sha256"),
        ("fold_manifest_sha256", "fold_manifest_sha256"),
        ("canonical_patient_order_sha256", "canonical_patient_order_sha256"),
        ("canonical_patient_label_sha256", "canonical_patient_label_sha256"),
        ("static_transform_sha256", "static_transform_sha256"),
        ("change_transform_sha256", "change_transform_sha256"),
        ("raw_target_file_sha256", "raw_target_file_sha256"),
        ("raw_targets_sha256", "raw_targets_sha256"),
    ):
        if set(current[key].astype(str)) != {str(summary.get(summary_key))}:
            raise AnalysisInputError(
                f"probe summary/selection {key} 不一致: {selection_path}"
            )
    for row in current.itertuples(index=False):
        part = predictions.loc[
            predictions["task"].eq(row.task) & predictions["cell"].eq(row.cell)
        ]
        if len(part) != predictions["patient_id"].nunique() or any(
            set(part[prediction_column].astype(str))
            != {str(getattr(row, selection_column))}
            for prediction_column, selection_column in (
                ("selected_alpha", "selected_alpha"),
                ("target_transform", "target_transform"),
                ("source_feature_sha256", "source_feature_sha256"),
                ("source_checkpoint_sha256", "source_checkpoint_sha256"),
                ("static_transform_sha256", "static_transform_sha256"),
                ("change_transform_sha256", "change_transform_sha256"),
                ("raw_targets_sha256", "raw_targets_sha256"),
            )
        ):
            raise AnalysisInputError(
                f"probe selection/prediction 不一致: {selection_path}/{row.cell}"
            )

    fold = int(_integer_values(current["fold"], f"{label}.fold")[0])
    raw_targets, static_specs, change_specs = _load_probe_target_contract(
        current, fold, feature
    )
    patient_ids = np.asarray(feature["patient_ids"]).astype(str)
    splits = np.asarray(feature["splits"]).astype(str)
    response = np.asarray(feature["response_state"], dtype=np.float64)
    for row in current.itertuples(index=False):
        task, cell = str(row.task), str(row.cell)

        def prepare(
            split: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            indices: list[int] = []
            standardized: list[float] = []
            natural: list[float] = []
            for index in np.flatnonzero(splits == split):
                target_std, target_natural, valid = _probe_target_value(
                    patient_ids[index],
                    task,
                    cell,
                    raw_targets,
                    static_specs,
                    change_specs,
                )
                if valid:
                    indices.append(int(index))
                    standardized.append(target_std)
                    natural.append(target_natural)
            selected_indices = np.asarray(indices, dtype=np.int64)
            matrix = _probe_cell_matrix(response, selected_indices, task, cell)
            return (
                selected_indices,
                matrix,
                np.asarray(standardized, dtype=np.float64),
                np.asarray(natural, dtype=np.float64),
            )

        train_indices, train_matrix, train_target, _ = prepare("train")
        val_indices, val_matrix, val_target, _ = prepare("val")
        test_indices, test_matrix, test_target, test_natural = prepare("test")
        part = predictions.loc[
            predictions["task"].eq(task) & predictions["cell"].eq(cell)
        ]
        if (
            not np.array_equal(
                part["patient_id"].astype(str).to_numpy(), patient_ids[test_indices]
            )
            or not np.allclose(
                _numeric(part["y_true"], f"{label}.{cell}.y_true"),
                test_natural,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.allclose(
                _numeric(
                    part["y_true_standardized"], f"{label}.{cell}.y_true_standardized"
                ),
                test_target,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise AnalysisInputError(
                f"probe raw target/test patient 逐行复算失败: {selection_path}/{cell}"
            )
        if (
            len(train_indices) != int(row.n_train)
            or len(val_indices) != int(row.n_val)
            or len(test_indices) != int(row.n_test)
        ):
            raise AnalysisInputError(
                f"probe raw target split counts 复算失败: {selection_path}/{cell}"
            )
        fitted_scaler = StandardScaler().fit(train_matrix)
        saved_mean = _json_finite_vector(
            row.feature_scaler_mean_json, f"{label}.{cell}.scaler_mean", 192
        )
        saved_scale = _json_finite_vector(
            row.feature_scaler_scale_json, f"{label}.{cell}.scaler_scale", 192
        )
        if not np.allclose(
            saved_mean, fitted_scaler.mean_, rtol=1e-12, atol=1e-12
        ) or not np.allclose(saved_scale, fitted_scaler.scale_, rtol=1e-12, atol=1e-12):
            raise AnalysisInputError(
                f"probe scaler 不能由 outer-train feature 复算: {selection_path}/{cell}"
            )
        coefficient = _json_finite_vector(
            row.ridge_coef_json, f"{label}.{cell}.ridge_coef", 192
        )
        intercept = float(row.ridge_intercept)

        def predict(matrix: np.ndarray) -> np.ndarray:
            return ((matrix - saved_mean) / saved_scale) @ coefficient + intercept

        validation_prediction = predict(val_matrix)
        validation_mse = float(mean_squared_error(val_target, validation_prediction))
        b0_standardized = float(np.mean(train_target))
        b0_validation_mse = float(
            mean_squared_error(val_target, np.full(val_target.shape, b0_standardized))
        )
        test_prediction_standardized = predict(test_matrix)
        test_prediction_natural = _probe_inverse_prediction(
            test_prediction_standardized, task, cell, static_specs, change_specs
        )
        b0_natural = float(
            _probe_inverse_prediction(
                np.asarray([b0_standardized]),
                task,
                cell,
                static_specs,
                change_specs,
            )[0]
        )
        if (
            not math.isclose(
                float(row.val_mse_standardized),
                validation_mse,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.train_target_mean_standardized),
                b0_standardized,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.b0_val_mse_standardized),
                b0_validation_mse,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not np.allclose(
                _numeric(
                    part["y_pred_standardized"],
                    f"{label}.{cell}.y_pred_standardized",
                ),
                test_prediction_standardized,
                rtol=1e-10,
                atol=1e-12,
            )
            or not np.allclose(
                _numeric(part["y_pred"], f"{label}.{cell}.y_pred"),
                test_prediction_natural,
                rtol=1e-10,
                atol=1e-12,
            )
            or not np.allclose(
                _numeric(
                    part["b0_prediction_standardized"],
                    f"{label}.{cell}.b0_prediction_standardized",
                ),
                b0_standardized,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.allclose(
                _numeric(part["b0_prediction"], f"{label}.{cell}.b0_prediction"),
                b0_natural,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise AnalysisInputError(
                f"probe scaler/Ridge/B0/prediction 独立复算失败: {selection_path}/{cell}"
            )
    if str(summary.get("prediction_file_sha256")) != file_sha256(prediction_path):
        raise AnalysisInputError(
            f"probe summary prediction SHA 未闭合: {selection_path}"
        )


def _validate_pcr_selection_protocol(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    predictions: pd.DataFrame,
    selection_path: Path,
    prediction_path: Path,
    expected: Mapping[str, str],
    feature: Mapping[str, np.ndarray],
) -> None:
    label = f"pCR selection {selection_path}"
    _require_exact_columns(frame, PCR_SELECTION_COLUMNS, label)
    if summary.get("prediction_columns") != list(PCR_PREDICTION_COLUMNS) or summary.get(
        "selection_columns"
    ) != list(PCR_SELECTION_COLUMNS):
        raise AnalysisInputError(f"pCR summary column schema 漂移: {selection_path}")
    if (
        int(summary.get("readout_rng_base_seed", -1)) != PCR_READOUT_SEED
        or summary.get("decision_points") != list(DECISION_POINTS)
        or dict(summary.get("feature_schemas", {})) != PCR_FEATURE_SCHEMAS
        or {
            key: int(value)
            for key, value in dict(summary.get("feature_dims", {})).items()
        }
        != PCR_FEATURE_DIMS
        or int(summary.get("test_prediction_rows", -1)) != len(predictions)
        or int(summary.get("test_patient_count", -1))
        != predictions["patient_id"].nunique()
        or str(summary.get("pcr_implementation_sha256")) != pcr_implementation_sha256()
        or str(summary.get("sklearn_version")) != sklearn.__version__
    ):
        raise AnalysisInputError(
            f"pCR summary protocol/count/implementation 漂移: {selection_path}"
        )
    expected_logistic = {
        "penalties": list(PCR_PENALTIES),
        "C_grid": [float(value) for value in PCR_C_GRID],
        "solver": LOGISTIC_SOLVER,
        "random_state_schedule": (
            f"{PCR_READOUT_SEED} + fold*100 + decision_index; independent of encoder seed_base"
        ),
        "class_weight": "balanced",
        "scaler": "StandardScaler fit on outer fold train only",
        "hyperparameter_selection": "outer fold validation AUROC/AUPRC only",
        "threshold_selection": "outer fold validation Youden J only",
    }
    if dict(summary.get("logistic", {})) != expected_logistic:
        raise AnalysisInputError(
            f"pCR summary Logistic protocol 漂移: {selection_path}"
        )
    expected_guards = {
        "feature_inputs": "frozen observed response_state only",
        "clinical_used": False,
        "treatment_used": False,
        "radiomics_used": False,
        "mask_geometry_used": False,
        "ground_truth_ftv_used": False,
        "predicted_ftv_or_ftv_head_used": False,
        "feature_scaler_fit_scope": "outer fold train only",
        "logistic_fit_scope": "outer fold train only",
        "penalty_C_selection_scope": "outer fold validation only",
        "threshold_selection_scope": "outer fold validation only",
        "test_feature_matrix_constructed_after_selection_lock": True,
        "test_predict_proba_calls_per_decision": 1,
        "test_prediction_guard_enforced": True,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_lambda_selection": False,
        "test_used_for_any_selection": False,
        "world_model_trained_or_finetuned": False,
    }
    if dict(summary.get("leakage_guards", {})) != expected_guards:
        raise AnalysisInputError(f"pCR summary leakage guards 漂移: {selection_path}")

    current = frame.copy()
    current["decision_point"] = current["decision_point"].map(_normalise_decision)
    if (
        len(current) != 3
        or current["decision_point"].duplicated().any()
        or set(current["decision_point"]) != set(DECISION_POINTS)
    ):
        raise AnalysisInputError(
            f"pCR selection 三个 decision points 不唯一: {selection_path}"
        )
    for point in DECISION_POINTS:
        row = current.loc[current["decision_point"].eq(point)].iloc[0]
        schema = PCR_FEATURE_SCHEMAS[point]
        schema_sha = hashlib.sha256(schema.encode("utf-8")).hexdigest()
        if (
            str(row["feature_schema"]) != schema
            or str(row["feature_schema_sha256"]) != schema_sha
            or int(row["feature_dim"]) != PCR_FEATURE_DIMS[point]
        ):
            raise AnalysisInputError(
                f"pCR selection feature schema 漂移: {selection_path}/{point}"
            )
    if set(_integer_values(current["feature_dim"], f"{label}.feature_dim")) != set(
        PCR_FEATURE_DIMS.values()
    ) or set(current["sklearn_version"].astype(str)) != {sklearn.__version__}:
        raise AnalysisInputError(
            f"pCR selection feature_dim/sklearn version 漂移: {selection_path}"
        )
    selected_penalty = current["selected_penalty"].astype(str).str.lower()
    selected_c = _numeric(current["selected_C"], f"{label}.selected_C")
    selected_threshold = _numeric(
        current["selected_threshold"], f"{label}.selected_threshold"
    )
    if (
        not set(selected_penalty).issubset(set(PCR_PENALTIES))
        or not np.isin(selected_c, np.asarray(PCR_C_GRID, dtype=float)).all()
        or not ((selected_threshold >= 0) & (selected_threshold <= 1)).all()
        or set(current["solver"].astype(str)) != {LOGISTIC_SOLVER}
        or set(current["class_weight"].astype(str)) != {"balanced"}
        or not np.equal(
            _integer_values(current["max_iter"], f"{label}.max_iter"),
            LOGISTIC_MAX_ITER,
        ).all()
        or not np.equal(_numeric(current["tol"], f"{label}.tol"), LOGISTIC_TOL).all()
    ):
        raise AnalysisInputError(
            f"pCR penalty/C/threshold/solver protocol 漂移: {selection_path}"
        )
    expected_rule = (
        "max validation AUROC; <=1e-12 tie max AUPRC; then smaller C; "
        "then penalty order l1,l2"
    )
    expected_threshold_rule = (
        "max validation Youden J; <=1e-12 tie closest to 0.5; then smaller threshold"
    )
    if set(current["selection_rule"].astype(str)) != {expected_rule} or set(
        current["threshold_tie_rule"].astype(str)
    ) != {expected_threshold_rule}:
        raise AnalysisInputError(
            f"pCR validation selection/tie rule 漂移: {selection_path}"
        )
    _finite(
        current,
        (
            "val_auroc",
            "val_auprc",
            "selected_threshold",
            "val_youden",
            "val_sensitivity",
            "val_specificity",
            "tol",
        ),
        label,
    )
    for column in ("val_auroc", "val_auprc", "val_sensitivity", "val_specificity"):
        values = _numeric(current[column], f"{label}.{column}")
        if not ((values >= 0) & (values <= 1)).all():
            raise AnalysisInputError(f"pCR {column} 超出 [0,1]: {selection_path}")
    youden = _numeric(current["val_youden"], f"{label}.val_youden")
    if not ((youden >= -1) & (youden <= 1)).all():
        raise AnalysisInputError(f"pCR val_youden 超出 [-1,1]: {selection_path}")
    counts = {
        name: _integer_values(current[name], f"{label}.{name}")
        for name in (
            "n_train",
            "n_val",
            "n_test",
            "train_positive",
            "val_positive",
            "test_positive",
            "feature_scaler_n_samples_seen",
        )
    }
    if (
        any((counts[name] <= 0).any() for name in ("n_train", "n_val", "n_test"))
        or not np.array_equal(
            counts["n_train"], counts["feature_scaler_n_samples_seen"]
        )
        or not np.equal(counts["n_test"], predictions["patient_id"].nunique()).all()
        or any(
            (counts[positive] < 0).any() or (counts[positive] > counts[total]).any()
            for positive, total in (
                ("train_positive", "n_train"),
                ("val_positive", "n_val"),
                ("test_positive", "n_test"),
            )
        )
    ):
        raise AnalysisInputError(
            f"pCR train/val/test/scaler/class counts 不闭合: {selection_path}"
        )
    penalty_order = {value: index for index, value in enumerate(PCR_PENALTIES)}
    expected_grid = {
        (penalty, float(c_value)) for penalty in PCR_PENALTIES for c_value in PCR_C_GRID
    }
    for row in current.itertuples(index=False):
        try:
            grid = json.loads(str(row.grid_validation_metrics_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AnalysisInputError(
                f"pCR grid validation evidence 非法: {selection_path}"
            ) from exc
        if not isinstance(grid, list) or len(grid) != len(expected_grid):
            raise AnalysisInputError(
                f"pCR penalty/C grid evidence 行数错误: {selection_path}"
            )
        candidates: list[tuple[float, float, float, int, str]] = []
        observed_grid: set[tuple[str, float]] = set()
        for candidate in grid:
            if not isinstance(candidate, dict):
                raise AnalysisInputError(
                    f"pCR grid candidate 非 object: {selection_path}"
                )
            penalty = str(candidate.get("penalty", "")).lower()
            try:
                c_value = float(candidate["C"])
                auroc = float(candidate["val_auroc"])
                auprc = float(candidate["val_auprc"])
                n_iter_raw = float(candidate["n_iter"])
                n_iter = int(n_iter_raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisInputError(
                    f"pCR grid candidate schema 非法: {selection_path}"
                ) from exc
            if (
                not np.isfinite((c_value, auroc, auprc)).all()
                or not (0 <= auroc <= 1 and 0 <= auprc <= 1)
                or penalty not in penalty_order
                or n_iter_raw != n_iter
                or n_iter < 0
                or n_iter > LOGISTIC_MAX_ITER
            ):
                raise AnalysisInputError(f"pCR grid candidate 值非法: {selection_path}")
            observed_grid.add((penalty, c_value))
            candidates.append((auroc, auprc, c_value, penalty_order[penalty], penalty))
        if observed_grid != expected_grid:
            raise AnalysisInputError(
                f"pCR penalty/C grid evidence 不完整: {selection_path}"
            )
        best_auroc = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] >= best_auroc - 1e-12]
        best_auprc = max(item[1] for item in tied)
        tied = [item for item in tied if item[1] >= best_auprc - 1e-12]
        chosen = min(tied, key=lambda item: (item[2], item[3]))
        if (
            str(row.selected_penalty).lower() != chosen[4]
            or not math.isclose(float(row.selected_C), chosen[2], rel_tol=0, abs_tol=0)
            or not math.isclose(
                float(row.val_auroc), chosen[0], rel_tol=0, abs_tol=1e-12
            )
            or not math.isclose(
                float(row.val_auprc), chosen[1], rel_tol=0, abs_tol=1e-12
            )
        ):
            raise AnalysisInputError(
                f"pCR validation-only penalty/C selection 不可复算: {selection_path}"
            )
        dimension = PCR_FEATURE_DIMS[str(row.decision_point)]
        _json_finite_vector(row.logistic_intercept_json, f"{label}.intercept", 1)
        _json_finite_vector(row.logistic_coef_json, f"{label}.coef", dimension)
        _json_finite_vector(
            row.feature_scaler_mean_json, f"{label}.scaler_mean", dimension
        )
        scale = _json_finite_vector(
            row.feature_scaler_scale_json, f"{label}.scaler_scale", dimension
        )
        if (scale <= 0).any():
            raise AnalysisInputError(f"pCR scaler scale 非正: {selection_path}")
    for column in (
        "feature_schema_sha256",
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
    ):
        _require_sha(current[column], f"{label}.{column}")
    if (
        set(current["source_feature_sha256"].astype(str))
        != {expected["feature_sha256"]}
        or set(current["source_checkpoint_sha256"].astype(str))
        != {expected["checkpoint_sha256"]}
        or set(current["feature_extractor_sha256"].astype(str))
        != {expected["feature_extractor_sha256"]}
        or set(current["canonical_patient_order_sha256"].astype(str))
        != {expected["canonical_patient_order_sha256"]}
        or set(current["canonical_patient_label_sha256"].astype(str))
        != {expected["canonical_patient_label_sha256"]}
        or set(current["fold_manifest_sha256"].astype(str))
        != {EXPECTED_FOLD_MANIFEST_SHA256}
    ):
        raise AnalysisInputError(
            f"pCR selection source/canonical hash 未闭合: {selection_path}"
        )
    for key in (
        "source_feature_sha256",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
    ):
        if set(current[key].astype(str)) != {str(summary.get(key))}:
            raise AnalysisInputError(
                f"pCR summary/selection {key} 不一致: {selection_path}"
            )
    for row in current.itertuples(index=False):
        part = predictions.loc[predictions["decision_point"].eq(row.decision_point)]
        scalar_pairs = (
            ("threshold", "selected_threshold"),
            ("penalty", "selected_penalty"),
            ("C", "selected_C"),
            ("val_auroc", "val_auroc"),
            ("val_auprc", "val_auprc"),
            ("val_youden", "val_youden"),
            ("feature_schema_sha256", "feature_schema_sha256"),
            ("source_feature_sha256", "source_feature_sha256"),
            ("source_checkpoint_sha256", "source_checkpoint_sha256"),
        )
        if len(part) != predictions["patient_id"].nunique() or any(
            set(part[prediction_column].astype(str))
            != {str(getattr(row, selection_column))}
            for prediction_column, selection_column in scalar_pairs
        ):
            raise AnalysisInputError(
                f"pCR selection/prediction 不一致: {selection_path}/{row.decision_point}"
            )

    patient_ids = np.asarray(feature["patient_ids"]).astype(str)
    splits = np.asarray(feature["splits"]).astype(str)
    labels = np.asarray(feature["label_pcr"], dtype=np.int64)
    response = np.asarray(feature["response_state"], dtype=np.float64)
    split_indices = {
        split: np.flatnonzero(splits == split) for split in ("train", "val", "test")
    }
    for row in current.itertuples(index=False):
        point = str(row.decision_point)
        dimension = PCR_FEATURE_DIMS[point]
        matrices = {
            split: _pcr_readout_matrix(response[indices], point)
            for split, indices in split_indices.items()
        }
        fitted_scaler = StandardScaler().fit(matrices["train"])
        saved_mean = _json_finite_vector(
            row.feature_scaler_mean_json, f"{label}.{point}.scaler_mean", dimension
        )
        saved_scale = _json_finite_vector(
            row.feature_scaler_scale_json, f"{label}.{point}.scaler_scale", dimension
        )
        if not np.allclose(
            saved_mean, fitted_scaler.mean_, rtol=1e-12, atol=1e-12
        ) or not np.allclose(saved_scale, fitted_scaler.scale_, rtol=1e-12, atol=1e-12):
            raise AnalysisInputError(
                f"pCR scaler 不能由 outer-train response_state 复算: {selection_path}/{point}"
            )
        coefficient = _json_finite_vector(
            row.logistic_coef_json, f"{label}.{point}.logistic_coef", dimension
        )
        intercept = _json_finite_vector(
            row.logistic_intercept_json, f"{label}.{point}.logistic_intercept", 1
        )[0]

        def probability(split: str) -> np.ndarray:
            scaled = (matrices[split] - saved_mean) / saved_scale
            return expit(scaled @ coefficient + intercept)

        validation_probability = probability("val")
        test_probability = probability("test")
        validation_labels = labels[split_indices["val"]]
        val_auroc = float(roc_auc_score(validation_labels, validation_probability))
        val_auprc = float(
            average_precision_score(validation_labels, validation_probability)
        )
        threshold, youden, sensitivity, specificity = _independent_youden(
            validation_labels, validation_probability
        )
        part = predictions.loc[predictions["decision_point"].eq(point)]
        expected_test_labels = labels[split_indices["test"]]
        expected_prediction = (test_probability >= threshold).astype(np.int64)
        exact_counts = {
            "n_train": len(split_indices["train"]),
            "n_val": len(split_indices["val"]),
            "n_test": len(split_indices["test"]),
            "train_positive": int(labels[split_indices["train"]].sum()),
            "val_positive": int(labels[split_indices["val"]].sum()),
            "test_positive": int(expected_test_labels.sum()),
            "feature_scaler_n_samples_seen": len(split_indices["train"]),
        }
        if any(
            int(getattr(row, name)) != value for name, value in exact_counts.items()
        ):
            raise AnalysisInputError(
                f"pCR canonical split/class/scaler counts 复算失败: {selection_path}/{point}"
            )
        if (
            not np.array_equal(
                part["patient_id"].astype(str).to_numpy(),
                patient_ids[split_indices["test"]],
            )
            or not np.array_equal(
                _integer_values(part["y_true"], f"{label}.{point}.y_true"),
                expected_test_labels,
            )
            or not np.allclose(
                _numeric(part["probability"], f"{label}.{point}.probability"),
                test_probability,
                rtol=1e-10,
                atol=1e-12,
            )
            or not np.array_equal(
                _integer_values(
                    part["predicted_label"], f"{label}.{point}.predicted_label"
                ),
                expected_prediction,
            )
            or not math.isclose(
                float(row.val_auroc), val_auroc, rel_tol=1e-10, abs_tol=1e-12
            )
            or not math.isclose(
                float(row.val_auprc), val_auprc, rel_tol=1e-10, abs_tol=1e-12
            )
            or not math.isclose(
                float(row.selected_threshold),
                threshold,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.val_youden), youden, rel_tol=1e-10, abs_tol=1e-12
            )
            or not math.isclose(
                float(row.val_sensitivity),
                sensitivity,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.val_specificity),
                specificity,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise AnalysisInputError(
                f"pCR scaler/logistic/threshold/test prediction 独立复算失败: {selection_path}/{point}"
            )
    if str(summary.get("prediction_file_sha256")) != file_sha256(prediction_path):
        raise AnalysisInputError(f"pCR summary prediction SHA 未闭合: {selection_path}")


def _audit_downstream_selections(
    config: AnalysisConfig,
    asset_hashes: Mapping[tuple[int, str, int], Mapping[str, str]],
    probe_predictions: pd.DataFrame,
    pcr_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """验证 50 FTV-only Ridge 与 50 pCR selection/summary/test-once provenance。"""

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for seed, model, fold in sorted(_expected_grid()):
        expected = asset_hashes[(seed, model, fold)]
        feature_path = Path(expected["feature_path"])
        if (
            not feature_path.is_file()
            or file_sha256(feature_path) != expected["feature_sha256"]
        ):
            raise AnalysisInputError(
                f"downstream independent recomputation feature 漂移: {feature_path}"
            )
        with np.load(feature_path, allow_pickle=False) as archive:
            feature = {
                name: archive[name].copy()
                for name in (
                    "patient_ids",
                    "splits",
                    "response_state",
                    "label_pcr",
                )
            }
        for kind in ("representation_probes", "pcr_readouts"):
            selection_path = _selection_path(
                config.metric_input_root, kind, seed, model, fold
            )
            summary_path = _summary_path(
                config.metric_input_root, kind, seed, model, fold
            )
            if not selection_path.is_file() or not summary_path.is_file():
                raise AnalysisInputError(
                    f"缺 downstream selection/summary: {selection_path} / {summary_path}"
                )
            frame = pd.read_csv(selection_path, low_memory=False)
            summary = _read_json(summary_path, f"{kind} summary")
            if int(summary.get("schema_version", -1)) != 2:
                raise AnalysisInputError(
                    f"downstream summary 必须是 schema-v2: {summary_path}"
                )
            prediction_path = _prediction_path(
                config.prediction_root, kind, seed, model, fold
            )
            if not prediction_path.is_file():
                raise AnalysisInputError(f"缺 downstream prediction: {prediction_path}")
            if (
                int(summary.get("seed_base", -1)) != seed
                or int(summary.get("effective_seed", -1)) != seed + fold
                or int(summary.get("fold", -1)) != fold
                or str(summary.get("model", "")).upper() != model
                or str(summary.get("selection_file_sha256"))
                != file_sha256(selection_path)
                or str(summary.get("prediction_file_sha256"))
                != file_sha256(prediction_path)
            ):
                raise AnalysisInputError(
                    f"downstream summary identity/live hash 错误: {summary_path}"
                )
            identity = (
                set(frame["model"].astype(str).str.upper()),
                set(_integer_values(frame["fold"], f"{kind}.fold")),
                set(_integer_values(frame["seed_base"], f"{kind}.seed_base")),
                set(_integer_values(frame["effective_seed"], f"{kind}.effective_seed")),
            )
            if identity != ({model}, {fold}, {seed}, {seed + fold}):
                raise AnalysisInputError(
                    f"downstream selection path/model/seed/fold 不一致: {selection_path}"
                )
            if kind == "representation_probes":
                false_flags = PROBE_FALSE_FLAGS
                true_flags = ("test_prediction_guard_enforced",)
                call_column = "test_predict_call_count"
                expected_cells = 7
                kind_label = "probe_selection"
                predictions = probe_predictions.loc[
                    probe_predictions["seed_base"].eq(seed)
                    & probe_predictions["model"].eq(model)
                    & probe_predictions["fold"].eq(fold)
                ]
                _validate_probe_selection_protocol(
                    frame,
                    summary,
                    predictions,
                    selection_path,
                    prediction_path,
                    expected,
                    feature,
                )
            else:
                points = frame["decision_point"].map(_normalise_decision)
                expected_rng = {
                    point: PCR_READOUT_SEED + fold * 100 + index
                    for index, point in enumerate(DECISION_POINTS)
                }
                for point, observed in zip(points, frame["random_state"], strict=True):
                    observed_rng = _integer_values(
                        pd.Series([observed]),
                        f"pCR selection {selection_path}.random_state",
                    )[0]
                    if int(observed_rng) != expected_rng[point]:
                        raise AnalysisInputError(
                            f"pCR readout RNG 漂移: {selection_path}/{point}"
                        )
                false_flags = PCR_FALSE_FLAGS
                true_flags = (
                    "test_feature_matrix_constructed_after_selection_lock",
                    "test_prediction_guard_enforced",
                )
                call_column = "test_predict_proba_call_count"
                expected_cells = 3
                kind_label = "pcr_selection"
                predictions = pcr_predictions.loc[
                    pcr_predictions["seed_base"].eq(seed)
                    & pcr_predictions["model"].eq(model)
                    & pcr_predictions["fold"].eq(fold)
                ]
                _validate_pcr_selection_protocol(
                    frame,
                    summary,
                    predictions,
                    selection_path,
                    prediction_path,
                    expected,
                    feature,
                )
            for column in false_flags:
                if _strict_bool(frame[column], f"{kind}.{column}").any():
                    raise AnalysisInputError(
                        f"downstream selection 使用 test: {selection_path}/{column}"
                    )
            for column in true_flags:
                if not _strict_bool(frame[column], f"{kind}.{column}").all():
                    raise AnalysisInputError(
                        f"downstream selection guard false: {selection_path}/{column}"
                    )
            if not np.equal(
                _integer_values(frame[call_column], f"{kind}.{call_column}"), 1
            ).all():
                raise AnalysisInputError(
                    f"downstream test call count 非 1: {selection_path}"
                )
            if set(frame["source_feature_sha256"].astype(str)) != {
                expected["feature_sha256"]
            }:
                raise AnalysisInputError(
                    f"selection source feature hash 未闭合: {selection_path}"
                )
            if set(frame["source_checkpoint_sha256"].astype(str)) != {
                expected["checkpoint_sha256"]
            }:
                raise AnalysisInputError(
                    f"selection source checkpoint hash 未闭合: {selection_path}"
                )
            rows.append(
                {
                    "kind": kind_label,
                    "seed_base": seed,
                    "model": model,
                    "fold": fold,
                    "cells": expected_cells,
                    "test_once": True,
                    "source_feature_sha256": expected["feature_sha256"],
                    "source_checkpoint_sha256": expected["checkpoint_sha256"],
                }
            )
            manifests.extend(
                (
                    _manifest_row(
                        kind_label,
                        selection_path,
                        seed_base=seed,
                        model=model,
                        fold=fold,
                        rows=len(frame),
                    ),
                    _manifest_row(
                        f"{kind_label}_summary",
                        summary_path,
                        seed_base=seed,
                        model=model,
                        fold=fold,
                    ),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(manifests)


def audit_training_assets(
    config: AnalysisConfig,
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """审计 50 checkpoint/history/selection/features，并构造公开聚合稳定性表。"""

    freeze = _validate_freeze_provenance()

    required_history = {
        "epoch",
        "seed_base",
        "fold",
        "effective_seed",
        "model",
        "lambda_ftv",
        "total_loss",
        "base_loss",
        "ftv_loss",
        "weighted_ftv_loss",
        "val_base_loss",
        "val_base_objective",
        "val_state_loss",
        "val_sigreg_loss",
        "val_ftv_metric",
        "val_grounded_patients",
        "val_representation_std",
        "representation_std",
        "is_selected_checkpoint",
    }
    stability_rows: list[dict[str, Any]] = []
    history_manifest: list[dict[str, Any]] = []
    selection_manifest: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    payloads: dict[tuple[int, str, int], dict[str, Any]] = {}
    assets: dict[tuple[int, str, int], dict[str, str]] = {}
    feature_fingerprints: dict[tuple[int, str, int], str] = {}
    for seed, model, fold in sorted(_expected_grid()):
        checkpoint = (
            config.checkpoint_root
            / f"seed_{seed}"
            / model.lower()
            / f"fold_{fold}"
            / "best.pt"
        )
        selection_path = checkpoint.parent / "selection.json"
        resolved_path = checkpoint.parent / "resolved_run.json"
        history_path = (
            config.history_root / f"seed_{seed}" / model.lower() / f"fold_{fold}.csv"
        )
        feature_dir = config.feature_root / f"seed_{seed}" / model / f"fold_{fold}"
        feature_path = feature_dir / "observed_features.npz"
        feature_metadata_path = feature_dir / "extraction_metadata.json"
        feature_fragment_path = feature_dir / "feature_manifest_fragment.csv"
        for path, label in (
            (checkpoint, "checkpoint"),
            (selection_path, "training selection"),
            (resolved_path, "resolved run"),
            (history_path, "history"),
            (feature_path, "feature"),
            (feature_metadata_path, "feature metadata"),
            (feature_fragment_path, "feature manifest fragment"),
        ):
            if not path.is_file():
                raise AnalysisInputError(f"缺正式 {label}: {path}")
        selection = _read_json(selection_path, "training selection")
        resolved = _read_json(resolved_path, "resolved run")
        history = pd.read_csv(history_path, low_memory=False)
        _require_columns(history, required_history, f"history {history_path}")
        if not (1 <= len(history) <= 12):
            raise AnalysisInputError(f"history epoch 行数不在 1..12: {history_path}")
        epochs = _integer_values(history["epoch"], f"history {history_path}.epoch")
        if history["epoch"].duplicated().any() or not np.array_equal(
            epochs, np.arange(1, len(history) + 1)
        ):
            raise AnalysisInputError(f"history epoch 重复: {history_path}")
        if (
            set(_integer_values(history["seed_base"], f"history {history_path}.seed"))
            != {seed}
            or set(_integer_values(history["fold"], f"history {history_path}.fold"))
            != {fold}
            or set(
                _integer_values(
                    history["effective_seed"], f"history {history_path}.effective_seed"
                )
            )
            != {seed + fold}
            or set(history["model"].astype(str).str.upper()) != {model}
        ):
            raise AnalysisInputError(
                f"history seed/model/fold contract 错误: {history_path}"
            )
        expected_lambda = 0.0 if model == "G1" else 0.25
        if not np.equal(
            _numeric(history["lambda_ftv"], f"history {history_path}.lambda_ftv"),
            expected_lambda,
        ).all():
            raise AnalysisInputError(f"history lambda_FTV 漂移: {history_path}")
        for boolean_column in ("noncollapse", "base_gate_pass", "checkpoint_eligible"):
            if boolean_column not in history:
                raise AnalysisInputError(
                    f"history 缺 selection evidence 布尔列: {history_path}/{boolean_column}"
                )
            _strict_bool(history[boolean_column], f"history.{boolean_column}")
        selected_mask = _strict_bool(
            history["is_selected_checkpoint"], "history.selected"
        )
        if int(selected_mask.sum()) != 1:
            raise AnalysisInputError(
                f"history 必须恰有一个 selected epoch: {history_path}"
            )
        selected_row = history.loc[selected_mask].iloc[0]
        if (
            int(selection.get("schema_version", -1)) != 1
            or int(selection.get("seed_base", -1)) != seed
            or int(selection.get("effective_seed", -1)) != seed + fold
            or int(selection.get("fold", -1)) != fold
            or str(selection.get("model_name", "")).upper() != model
            or int(selection.get("selected_epoch", -1)) != int(selected_row["epoch"])
            or selection.get("test_data_used") is not False
        ):
            raise AnalysisInputError(
                f"training selection contract 错误: {selection_path}"
            )
        expected_selection_rule = (
            "non-collapse epoch with minimum validation normalized next-state loss"
            if model == "G1"
            else "non-collapse and <=5% paired-baseline normalized next-state loss degradation; minimum validation FTV loss"
        )
        expected_fallback_rule = "minimum normalized-next-state gate violation, then minimum validation FTV loss among non-collapse finite epochs"
        mode = str(selection.get("selection_mode", ""))
        baseline_contract = dict(selection.get("baseline_selection_contract", {}))
        if (
            selection.get("selection_rule") != expected_selection_rule
            or selection.get("fallback_rule") != expected_fallback_rule
            or mode not in {"primary", "fallback_base_gate_failed"}
            or (model == "G1" and mode != "primary")
        ):
            raise AnalysisInputError(
                f"training checkpoint selection rule/mode 漂移: {selection_path}"
            )
        allowed_base_loss = (
            math.inf
            if model == "G1"
            else float(baseline_contract.get("allowed_val_base_loss", math.nan))
        )
        representation_std = pd.to_numeric(
            history["representation_std"], errors="coerce"
        ).to_numpy(dtype=float)
        val_base_all = pd.to_numeric(
            history["val_base_loss"], errors="coerce"
        ).to_numpy(dtype=float)
        val_ftv_all = pd.to_numeric(
            history["val_ftv_metric"], errors="coerce"
        ).to_numpy(dtype=float)
        grounded_all = pd.to_numeric(
            history["val_grounded_patients"], errors="coerce"
        ).to_numpy(dtype=float)
        expected_noncollapse = np.isfinite(representation_std) & (
            representation_std >= 0.05
        )
        expected_base_gate = (
            np.ones(len(history), dtype=bool)
            if model == "G1"
            else np.isfinite(val_base_all) & (val_base_all <= allowed_base_loss)
        )
        expected_ftv_finite = np.isfinite(val_ftv_all) & (grounded_all > 0)
        expected_eligible = expected_noncollapse & expected_base_gate
        if model == "G3":
            expected_eligible &= expected_ftv_finite
        if (
            not np.array_equal(
                _strict_bool(history["noncollapse"], "history.noncollapse"),
                expected_noncollapse,
            )
            or not np.array_equal(
                _strict_bool(history["base_gate_pass"], "history.base_gate_pass"),
                expected_base_gate,
            )
            or not np.array_equal(
                _strict_bool(
                    history["checkpoint_eligible"], "history.checkpoint_eligible"
                ),
                expected_eligible,
            )
        ):
            raise AnalysisInputError(
                f"history noncollapse/base-gate/eligible 不能机械复算: {history_path}"
            )
        best_metric = math.inf
        fallback_metric = (math.inf, math.inf)
        has_primary = False
        stale = 0
        stop_index: int | None = None
        for index in range(len(history)):
            metric = val_base_all[index] if model == "G1" else val_ftv_all[index]
            improved = False
            if expected_eligible[index] and metric < best_metric:
                best_metric = float(metric)
                has_primary = True
                improved = True
            if expected_noncollapse[index] and expected_ftv_finite[index]:
                violation = (
                    max(0.0, float(val_base_all[index]) - allowed_base_loss)
                    if model == "G3"
                    else 0.0
                )
                candidate = (violation, float(metric))
                if candidate < fallback_metric:
                    fallback_metric = candidate
                    if not has_primary:
                        improved = True
            stale = 0 if improved else stale + 1
            if stale >= 4:
                stop_index = index
                break
        if stop_index is not None and stop_index != len(history) - 1:
            raise AnalysisInputError(
                f"history 含 early-stop 后额外 epoch: {history_path}"
            )
        if stop_index is None and len(history) != 12:
            raise AnalysisInputError(
                f"history 在 patience/12 epochs 前被截断: {history_path}"
            )
        if not math.isclose(
            float(selection.get("selected_validation_ftv_loss", math.nan)),
            float(selected_row["val_ftv_metric"]),
            rel_tol=0,
            abs_tol=1e-10,
        ) or not math.isclose(
            float(selection.get("selected_representation_std", math.nan)),
            float(selected_row["representation_std"]),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise AnalysisInputError(
                f"selection/history selected FTV/std 不一致: {selection_path}"
            )
        eligible = _strict_bool(
            history["checkpoint_eligible"], f"history {history_path}.eligible"
        ).to_numpy()
        if mode == "primary":
            if not eligible.any():
                raise AnalysisInputError(
                    f"primary selection 但无 eligible epoch: {selection_path}"
                )
            metric_column = "val_base_loss" if model == "G1" else "val_ftv_metric"
            metric = pd.to_numeric(history[metric_column], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(metric[eligible]).all():
                raise AnalysisInputError(
                    f"eligible epoch 的 validation metric 非有限: {history_path}/{metric_column}"
                )
            expected_index = np.flatnonzero(eligible)[int(np.argmin(metric[eligible]))]
            if int(selected_row["epoch"]) != int(history.iloc[expected_index]["epoch"]):
                raise AnalysisInputError(
                    f"selected epoch 不能由冻结 validation rule 复算: {selection_path}"
                )
        else:
            allowed = float(baseline_contract.get("allowed_val_base_loss", math.nan))
            noncollapse = _strict_bool(
                history["noncollapse"], f"history {history_path}.noncollapse"
            ).to_numpy()
            val_ftv = pd.to_numeric(
                history["val_ftv_metric"], errors="coerce"
            ).to_numpy(dtype=float)
            val_base = pd.to_numeric(
                history["val_base_loss"], errors="coerce"
            ).to_numpy(dtype=float)
            grounded = pd.to_numeric(
                history["val_grounded_patients"], errors="coerce"
            ).to_numpy(dtype=float)
            candidates = np.flatnonzero(
                noncollapse
                & np.isfinite(val_ftv)
                & np.isfinite(val_base)
                & np.isfinite(grounded)
                & (grounded > 0)
            )
            if not np.isfinite(allowed) or not len(candidates):
                raise AnalysisInputError(
                    f"G3 fallback evidence 不完整: {selection_path}"
                )
            expected_index = min(
                candidates,
                key=lambda index: (
                    max(0.0, float(val_base[index]) - allowed),
                    float(val_ftv[index]),
                ),
            )
            if int(selected_row["epoch"]) != int(history.iloc[expected_index]["epoch"]):
                raise AnalysisInputError(
                    f"G3 fallback epoch 不能由冻结 validation rule 复算: {selection_path}"
                )
        selection_epochs = selection.get("epochs")
        if not isinstance(selection_epochs, list) or len(selection_epochs) != len(
            history
        ):
            raise AnalysisInputError(
                f"selection epoch evidence 行数不闭合: {selection_path}"
            )
        epoch_pairs = (
            ("epoch", "epoch"),
            ("seed_base", "seed_base"),
            ("fold", "fold"),
            ("effective_seed", "effective_seed"),
            ("val_base_loss", "val_base_loss"),
            ("val_base_objective", "val_base_objective"),
            ("val_state_loss", "val_state_loss"),
            ("val_sigreg_loss", "val_sigreg_loss"),
            ("val_ftv_loss", "val_ftv_metric"),
            ("val_representation_std", "representation_std"),
        )
        for index, epoch_payload in enumerate(selection_epochs):
            if not isinstance(epoch_payload, dict):
                raise AnalysisInputError(
                    f"selection epoch evidence 非 object: {selection_path}/{index}"
                )
            for selection_key, history_key in epoch_pairs:
                if not math.isclose(
                    float(epoch_payload.get(selection_key, math.nan)),
                    float(history.iloc[index][history_key]),
                    rel_tol=0,
                    abs_tol=1e-10,
                ):
                    raise AnalysisInputError(
                        f"selection/history epoch evidence 漂移: {selection_path}/{index}/{selection_key}"
                    )
            for selection_key, expected_value in (
                ("noncollapse", expected_noncollapse[index]),
                ("base_gate_pass", expected_base_gate[index]),
                ("eligible", expected_eligible[index]),
            ):
                if epoch_payload.get(selection_key) is not bool(expected_value):
                    raise AnalysisInputError(
                        f"selection epoch boolean evidence 漂移: {selection_path}/{index}/{selection_key}"
                    )
        if (
            int(resolved.get("seed_base", -1)) != seed
            or int(resolved.get("effective_seed", -1)) != seed + fold
            or str(resolved.get("model_name", "")).upper() != model
            or int(resolved.get("fold", -1)) != fold
            or int(resolved.get("seed", -1)) != seed + fold
            or resolved.get("smoke_patients") is not None
            or int(resolved.get("epochs_requested", -1)) != 12
            or str(resolved.get("experiment_plan_sha256")) != freeze["plan_sha256"]
            or str(resolved.get("source_commit")) != freeze["source_commit"]
            or str(resolved.get("implementation_sha256"))
            != freeze["training_implementation_sha256"]
        ):
            raise AnalysisInputError(
                f"resolved run seed/model contract 错误: {resolved_path}"
            )
        if not math.isclose(
            float(resolved.get("lambda_ftv", math.nan)), expected_lambda, abs_tol=0.0
        ):
            raise AnalysisInputError(
                f"lambda_FTV 未锁定为 {expected_lambda}: {resolved_path}"
            )
        _finite(
            pd.DataFrame([selected_row]),
            (
                "total_loss",
                "base_loss",
                "ftv_loss",
                "weighted_ftv_loss",
                "val_base_loss",
                "val_ftv_metric",
                "representation_std",
            ),
            f"selected history {history_path}",
        )
        if not math.isclose(
            float(selection["selected_validation_base_loss"]),
            float(selected_row["val_base_loss"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise AnalysisInputError(
                f"selection/history selected base loss 不一致: {selection_path}"
            )
        checkpoint_sha = file_sha256(checkpoint)
        checkpoint_contract: dict[str, Any] = {}
        if config.audit_checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise AnalysisInputError(f"checkpoint 顶层非 mapping: {checkpoint}")
            tensors_finite, tensor_count = _all_tensors_finite(payload)
        else:
            # 只供轻量 synthetic harness；正式 CLI 永远启用 checkpoint audit。
            payload = {}
            tensors_finite, tensor_count = True, 0
        if config.audit_checkpoints:
            if (
                int(payload.get("schema_version", -1)) != 2
                or payload.get("finalized") is not True
                or int(payload.get("seed_base", -1)) != seed
                or int(payload.get("effective_seed", -1)) != seed + fold
                or int(payload.get("fold", -1)) != fold
                or str(payload.get("model_name", "")).upper() != model
                or int(payload.get("epoch", -1)) != int(selected_row["epoch"])
                or str(payload.get("implementation_sha256"))
                != freeze["training_implementation_sha256"]
                or str(payload.get("source_commit")) != freeze["source_commit"]
                or str(payload.get("resolved_config_sha256", "")) == ""
            ):
                raise AnalysisInputError(
                    f"checkpoint finalized/seed/model contract 错误: {checkpoint}"
                )
            if str(payload.get("history_sha256")) != file_sha256(history_path) or str(
                payload.get("selection_sha256")
            ) != file_sha256(selection_path):
                raise AnalysisInputError(
                    f"checkpoint history/selection SHA 未闭合: {checkpoint}"
                )
            if str(payload.get("plan_sha256")) != freeze["plan_sha256"]:
                raise AnalysisInputError(f"checkpoint plan SHA 漂移: {checkpoint}")
            data_contract = dict(payload.get("data_contract", {}))
            if (
                str(data_contract.get("fold_manifest_sha256"))
                != EXPECTED_FOLD_MANIFEST_SHA256
            ):
                raise AnalysisInputError(
                    f"checkpoint fold manifest SHA 漂移: {checkpoint}"
                )
            _require_live_declared_file(
                data_contract.get("fold_manifest"),
                data_contract.get("fold_manifest_sha256"),
                f"checkpoint fold manifest {checkpoint}",
            )
            expected_transform_path = (
                EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
            ).resolve()
            if (
                Path(str(payload.get("ftv_transform_path", ""))).resolve()
                != expected_transform_path
            ):
                raise AnalysisInputError(
                    f"checkpoint FTV transform path 漂移: {checkpoint}"
                )
            _require_live_declared_file(
                expected_transform_path,
                payload.get("ftv_transform_sha256"),
                f"checkpoint FTV transform {checkpoint}",
            )
            payload_lambda = float(
                dict(payload.get("loss_config", {})).get("lambda_ftv", math.nan)
            )
            if not math.isclose(payload_lambda, expected_lambda, abs_tol=0.0):
                raise AnalysisInputError(f"checkpoint lambda_FTV 漂移: {checkpoint}")
            contract = dict(payload.get("architecture_contract", {}))
            expected_contract = {
                "schema_version": 1,
                "model_name": model,
                "backbone_input": "DCE7",
                "image_channels": 7,
                "first_conv_in_channels": 7,
                "roi_mask_backbone_input": False,
                "pooling": "gap",
                "roi_mask_use": "absent",
                "empty_roi_behavior": None,
                "observed_response_state": "online_preprojector_r",
                "response_dim": 192,
                "jepa_state": "projector(r)",
                "transition": "M0_direct_next_state_causal_transformer",
                "ftv_head": None if model == "G1" else "Linear(response_dim,1)",
                "ftv_is_forward_input": False,
                "forbidden_inputs_absent": [
                    "clinical",
                    "treatment",
                    "radiomics",
                    "mask_geometry",
                    "voxel_count",
                    "explicit_volume",
                ],
            }
            if contract != expected_contract:
                raise AnalysisInputError(
                    f"checkpoint architecture/transition contract 漂移: {checkpoint}"
                )
            expected_model_config = {
                "model_name": model,
                "image_channels": 7,
                "pooling": "gap",
                "direct_ftv_grounding": model == "G3",
                "base_channels": 16,
                "latent_dim": 192,
                "predictor_depth": 3,
                "predictor_heads": 4,
                "predictor_mlp_dim": 512,
                "dropout": 0.1,
            }
            expected_train_config = {
                "seed": seed,
                "batch_size": 32,
                "workers": 4,
                "epochs": 12,
                "patience": 4,
                "learning_rate": 5e-5,
                "weight_decay": 1e-4,
                "ema_momentum": 0.996,
                "max_grad_norm": 5.0,
                "min_representation_std": 0.05,
                "deterministic_algorithms": False,
            }
            expected_loss_config = {
                "lambda_ftv": expected_lambda,
                "sigreg": 0.09,
                "sigreg_projections": 256,
                "step_weights": [2.0, 1.0, 0.5],
            }
            if (
                dict(payload.get("model_config", {})) != expected_model_config
                or dict(payload.get("train_config", {})) != expected_train_config
                or dict(payload.get("loss_config", {})) != expected_loss_config
            ):
                raise AnalysisInputError(
                    f"checkpoint model/train/loss frozen config 漂移: {checkpoint}"
                )
            splits_payload = payload.get("splits")
            split_hashes = dict(payload.get("split_hashes", {}))
            if not isinstance(splits_payload, Mapping):
                raise AnalysisInputError(f"checkpoint 缺 splits: {checkpoint}")
            splits = {
                name: [str(value) for value in splits_payload.get(name, [])]
                for name in ("train", "val", "test", "pretrain_train")
            }
            expected_primary_counts = {
                "train": 525 if fold <= 2 else 526,
                "val": 121,
                "test": 162 if fold <= 2 else 161,
            }
            if (
                any(
                    len(splits[name]) != count
                    for name, count in expected_primary_counts.items()
                )
                or len(splits["pretrain_train"])
                != expected_primary_counts["train"] + 156
                or not set(splits["train"]).issubset(set(splits["pretrain_train"]))
                or len(set(splits["pretrain_train"]) - set(splits["train"])) != 156
                or bool(
                    (set(splits["pretrain_train"]) - set(splits["train"]))
                    & (set(splits["val"]) | set(splits["test"]))
                )
                or any(len(values) != len(set(values)) for values in splits.values())
                or any(
                    set(splits[left]).intersection(splits[right])
                    for index, left in enumerate(("train", "val", "test"))
                    for right in ("train", "val", "test")[index + 1 :]
                )
                or any(
                    str(split_hashes.get(name)) != patient_hash(values)
                    for name, values in splits.items()
                )
                or str(data_contract.get("train_patient_hash"))
                != patient_hash(splits["train"])
                or str(data_contract.get("val_patient_hash"))
                != patient_hash(splits["val"])
                or str(data_contract.get("test_patient_hash"))
                != patient_hash(splits["test"])
                or str(data_contract.get("extra_pretrain_patient_hash"))
                != patient_hash(set(splits["pretrain_train"]) - set(splits["train"]))
                or not SHA256_VALUE.fullmatch(
                    str(data_contract.get("raw_ftv_sha256", "")).lower()
                )
                or str(data_contract.get("raw_ftv_sha256")) != EXPECTED_RAW_FTV_SHA256
            ):
                raise AnalysisInputError(
                    f"checkpoint canonical split closure 失败: {checkpoint}"
                )
            runtime = dict(payload.get("runtime", {}))
            determinism = dict(payload.get("determinism", {}))
            expected_determinism = {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "seed": seed + fold,
                "shared_head_rng_isolation": True,
                "fixed_patient_order_seed": seed + fold,
                "no_random_augmentation": True,
                "cross_hardware_bitwise_reproducibility_claimed": False,
            }
            if (
                runtime.get("seed_base") != seed
                or runtime.get("fold") != fold
                or runtime.get("effective_seed") != seed + fold
                or runtime.get("seed") != seed + fold
                or runtime.get("smoke") is not False
                or runtime.get("smoke_patients_requested") is not None
                or runtime.get("effective_train_ids") != splits["pretrain_train"]
                or runtime.get("effective_validation_ids") != splits["val"]
                or runtime.get("transform_fit_ids") != splits["train"]
                or str(runtime.get("effective_train_patient_hash"))
                != patient_hash(splits["pretrain_train"])
                or str(runtime.get("effective_validation_patient_hash"))
                != patient_hash(splits["val"])
                or determinism != expected_determinism
                or data_contract.get("backbone_tensor") != "x[:, :7]"
                or data_contract.get("roi_mask_tensor") != "x[:, 7:8] kept separate"
            ):
                raise AnalysisInputError(
                    f"checkpoint runtime/patient-order/determinism contract 漂移: {checkpoint}"
                )
            git_payload = dict(payload.get("git", {}))
            if git_payload.get("branch") != EXPECTED_BRANCH:
                raise AnalysisInputError(f"checkpoint Git branch 漂移: {checkpoint}")
            selected_metrics = dict(payload.get("selected_epoch_metrics", {}))
            for payload_key, history_key in (
                ("epoch", "epoch"),
                ("val_base_loss", "val_base_loss"),
                ("val_ftv_loss", "val_ftv_metric"),
                ("val_representation_std", "representation_std"),
            ):
                if not math.isclose(
                    float(selected_metrics.get(payload_key, math.nan)),
                    float(selected_row[history_key]),
                    rel_tol=0,
                    abs_tol=1e-10,
                ):
                    raise AnalysisInputError(
                        f"checkpoint selected metrics/history 不一致: {checkpoint}/{payload_key}"
                    )
            if (
                dict(payload.get("baseline_selection_contract", {}))
                != dict(selection.get("baseline_selection_contract", {}))
                or dict(resolved.get("baseline_selection_contract", {}))
                != dict(selection.get("baseline_selection_contract", {}))
                or dict(resolved.get("model_config", {})) != expected_model_config
                or dict(resolved.get("architecture_contract", {})) != contract
                or str(resolved.get("shared_initialization_sha256"))
                != str(payload.get("shared_initialization_sha256"))
                or str(resolved.get("ftv_transform_sha256"))
                != str(payload.get("ftv_transform_sha256"))
            ):
                raise AnalysisInputError(
                    f"resolved/selection/checkpoint provenance 不闭合: {checkpoint}"
                )
            checkpoint_contract = {
                name: payload.get(name)
                for name in (
                    "implementation_sha256",
                    "source_commit",
                    "resolved_config_sha256",
                    "shared_initialization_sha256",
                    "split_hashes",
                    "ftv_transform_sha256",
                    "data_contract",
                    "model_config",
                    "train_config",
                    "loss_config",
                    "baseline_selection_contract",
                    "architecture_contract",
                    "runtime",
                    "determinism",
                )
            }
            payloads[(seed, model, fold)] = checkpoint_contract
            del payload
        feature_sha = file_sha256(feature_path)
        metadata = _read_json(feature_metadata_path, "feature metadata")
        if int(metadata.get("schema_version", -1)) != 2:
            raise AnalysisInputError(
                f"feature metadata 必须是 schema-v2: {feature_metadata_path}"
            )
        if (
            str(metadata.get("feature_file_sha256")) != feature_sha
            or str(metadata.get("checkpoint_sha256")) != checkpoint_sha
        ):
            raise AnalysisInputError(f"feature/checkpoint SHA 未闭合: {feature_dir}")
        if (
            int(metadata.get("seed_base", -1)) != seed
            or int(metadata.get("effective_seed", -1)) != seed + fold
            or int(metadata.get("fold", -1)) != fold
            or str(metadata.get("model", "")).upper() != model
        ):
            raise AnalysisInputError(
                f"feature metadata seed/model/fold contract 错误: {feature_dir}"
            )
        if (
            str(metadata.get("extractor_sha256")) != extraction_implementation_sha256()
            or str(metadata.get("checkpoint_plan_sha256")) != freeze["plan_sha256"]
            or str(metadata.get("checkpoint_training_implementation_sha256"))
            != freeze["training_implementation_sha256"]
            or str(metadata.get("checkpoint_source_commit")) != freeze["source_commit"]
            or str(metadata.get("checkpoint_resolved_config_sha256"))
            != str(checkpoint_contract.get("resolved_config_sha256"))
            or str(metadata.get("fold_manifest_sha256"))
            != EXPECTED_FOLD_MANIFEST_SHA256
            or metadata.get("max_patients_per_split") is not None
            or metadata.get("measurement_targets_read_during_extraction") is not False
            or metadata.get(
                "pcr_labels_attached_from_locked_manifest_for_downstream_only"
            )
            is not True
            or metadata.get("world_model_trained_or_finetuned") is not False
            or metadata.get("ftv_head_loaded_but_not_called") is not (model == "G3")
        ):
            raise AnalysisInputError(
                f"feature metadata provenance/guard 漂移: {feature_dir}"
            )
        fold_manifest_value = metadata.get("fold_manifest")
        if not isinstance(fold_manifest_value, str) or not fold_manifest_value:
            raise AnalysisInputError(
                f"feature metadata 缺 canonical fold manifest: {feature_dir}"
            )
        with np.load(feature_path, allow_pickle=False) as archive:
            required_arrays = {
                "patient_ids",
                "splits",
                "response_state",
                "timepoints",
                "model",
                "seed_base",
                "fold",
                "effective_seed",
                "label_pcr",
            }
            if missing := required_arrays.difference(archive.files):
                raise AnalysisInputError(
                    f"feature NPZ 缺字段 {sorted(missing)}: {feature_path}"
                )
            try:
                canonical_evidence = validate_feature_against_canonical(
                    archive,
                    fold_manifest=Path(fold_manifest_value),
                    fold=fold,
                    max_patients_per_split=None,
                )
            except (
                FileNotFoundError,
                TypeError,
                ValueError,
                FloatingPointError,
            ) as exc:
                raise AnalysisInputError(
                    f"feature 未逐行闭合 canonical manifest: {feature_path}: {exc}"
                ) from exc
            patient_ids = archive["patient_ids"].astype(str)
            splits = archive["splits"].astype(str)
            response = archive["response_state"]
            labels = archive["label_pcr"]
            if (
                response.shape != (808, 4, 192)
                or response.dtype != np.float32
                or not np.isfinite(response).all()
            ):
                raise AnalysisInputError(
                    f"feature response_state 非 finite float32 [808,4,192]: {feature_path}"
                )
            if (
                len(patient_ids) != 808
                or len(set(patient_ids)) != 808
                or splits.shape != (808,)
                or labels.shape != (808,)
            ):
                raise AnalysisInputError(
                    f"feature patient/split/label closure 错误: {feature_path}"
                )
            if tuple(archive["timepoints"].astype(str)) != TIMEPOINTS:
                raise AnalysisInputError(
                    f"feature timepoint order 漂移: {feature_path}"
                )
            if (
                str(archive["model"].reshape(()).item()).upper() != model
                or int(archive["fold"].reshape(()).item()) != fold
                or int(archive["seed_base"].reshape(()).item()) != seed
                or int(archive["effective_seed"].reshape(()).item()) != seed + fold
            ):
                raise AnalysisInputError(
                    f"feature model/seed/fold contract 错误: {feature_path}"
                )
            feature_fingerprints[(seed, model, fold)] = _hash_ordered_rows(
                patient_ids, splits, labels
            )
            patient_ids_copy = patient_ids.copy()
            splits_copy = splits.copy()
            labels_copy = labels.astype(np.int64, copy=True)
        canonical_test_mask = splits_copy == "test"
        canonical_test_ids = patient_ids_copy[canonical_test_mask]
        canonical_test_labels = labels_copy[canonical_test_mask]
        pcr_part = pcr.loc[
            pcr["seed_base"].eq(seed) & pcr["model"].eq(model) & pcr["fold"].eq(fold)
        ]
        for point in DECISION_POINTS:
            point_part = pcr_part.loc[pcr_part["decision_point"].eq(point)]
            if not np.array_equal(
                point_part["patient_id"].astype(str).to_numpy(), canonical_test_ids
            ) or not np.array_equal(
                _integer_values(
                    point_part["y_true"], f"pCR {seed}/{model}/{fold}/{point}"
                ),
                canonical_test_labels,
            ):
                raise AnalysisInputError(
                    f"pCR patient/order/label 未闭合 feature canonical test: {seed}/{model}/{fold}/{point}"
                )
        probe_part = probe.loc[
            probe["seed_base"].eq(seed)
            & probe["model"].eq(model)
            & probe["fold"].eq(fold)
        ]
        allowed_test_ids = set(canonical_test_ids.tolist())
        if not set(probe_part["patient_id"].astype(str)).issubset(allowed_test_ids):
            raise AnalysisInputError(
                f"probe 含非 canonical outer-test patient: {seed}/{model}/{fold}"
            )
        expected_split_counts = {
            "train": 525 if fold <= 2 else 526,
            "val": 121,
            "test": 162 if fold <= 2 else 161,
        }
        expected_architecture_sha = hashlib.sha256(
            json.dumps(
                checkpoint_contract.get("architecture_contract", {}),
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        coverage = dict(metadata.get("coverage", {}))
        if (
            metadata.get("patient_count") != 808
            or metadata.get("feature_shape") != [808, 4, 192]
            or metadata.get("feature_dtype") != "float32"
            or dict(metadata.get("split_counts", {})) != expected_split_counts
            or str(metadata.get("architecture_contract_sha256"))
            != expected_architecture_sha
            or str(metadata.get("manifest_file_sha256"))
            != file_sha256(feature_fragment_path)
            or metadata.get("canonical_manifest_rows_verified") is not True
            or metadata.get("canonical_label_rows_verified") is not True
            or str(metadata.get("canonical_patient_order_sha256"))
            != str(canonical_evidence["canonical_patient_order_sha256"])
            or str(metadata.get("canonical_patient_label_sha256"))
            != str(canonical_evidence["canonical_patient_label_sha256"])
            or coverage
            != {
                "expected_primary_patients": 808,
                "observed_primary_patients": 808,
                "all_four_visits_present": True,
                "response_rows_finite": True,
                "patient_ids_unique": True,
                "formal_complete": True,
            }
        ):
            raise AnalysisInputError(
                f"feature metadata shape/canonical/coverage 漂移: {feature_dir}"
            )
        fragment = pd.read_csv(feature_fragment_path, low_memory=False)
        _require_columns(
            fragment,
            (
                "patient_id",
                "seed_base",
                "fold",
                "effective_seed",
                "model",
                "feature_file_sha256",
                "source_checkpoint_sha256",
                "split",
                "patient_index",
                "label_pcr",
                "visits",
                "representation",
                "feature_dim",
                "fold_manifest_sha256",
                "canonical_patient_order_sha256",
                "canonical_patient_label_sha256",
                "extractor_sha256",
            ),
            f"feature manifest {feature_fragment_path}",
        )
        if (
            len(fragment) != 808
            or fragment["patient_id"].astype(str).duplicated().any()
            or not np.array_equal(
                fragment["patient_id"].astype(str).to_numpy(), patient_ids_copy
            )
            or not np.array_equal(fragment["split"].astype(str).to_numpy(), splits_copy)
            or not np.array_equal(
                _integer_values(fragment["label_pcr"], "feature fragment.label_pcr"),
                labels_copy,
            )
            or not np.array_equal(
                _integer_values(
                    fragment["patient_index"], "feature fragment.patient_index"
                ),
                np.arange(808),
            )
        ):
            raise AnalysisInputError(
                f"feature manifest patient coverage 错误: {feature_fragment_path}"
            )
        if (
            set(_integer_values(fragment["seed_base"], "feature fragment.seed_base"))
            != {seed}
            or set(
                _integer_values(
                    fragment["effective_seed"], "feature fragment.effective_seed"
                )
            )
            != {seed + fold}
            or set(_integer_values(fragment["fold"], "feature fragment.fold")) != {fold}
            or set(fragment["model"].astype(str).str.upper()) != {model}
        ):
            raise AnalysisInputError(
                f"feature manifest seed/model/fold contract 错误: {feature_fragment_path}"
            )
        if set(fragment["feature_file_sha256"].astype(str)) != {feature_sha} or set(
            fragment["source_checkpoint_sha256"].astype(str)
        ) != {checkpoint_sha}:
            raise AnalysisInputError(
                f"feature manifest SHA 未闭合: {feature_fragment_path}"
            )
        if (
            set(_integer_values(fragment["visits"], "feature fragment.visits")) != {4}
            or set(
                _integer_values(fragment["feature_dim"], "feature fragment.feature_dim")
            )
            != {192}
            or set(fragment["representation"].astype(str)) != {"response_state"}
            or set(fragment["fold_manifest_sha256"].astype(str))
            != {EXPECTED_FOLD_MANIFEST_SHA256}
            or set(fragment["canonical_patient_order_sha256"].astype(str))
            != {canonical_evidence["canonical_patient_order_sha256"]}
            or set(fragment["canonical_patient_label_sha256"].astype(str))
            != {canonical_evidence["canonical_patient_label_sha256"]}
            or set(fragment["extractor_sha256"].astype(str))
            != {extraction_implementation_sha256()}
        ):
            raise AnalysisInputError(
                f"feature manifest representation/canonical protocol 漂移: {feature_fragment_path}"
            )
        assets[(seed, model, fold)] = {
            "checkpoint_sha256": checkpoint_sha,
            "feature_sha256": feature_sha,
            "feature_path": str(feature_path.resolve()),
            "selected_val_base_loss": str(float(selected_row["val_base_loss"])),
            "feature_extractor_sha256": str(metadata["extractor_sha256"]),
            "canonical_patient_order_sha256": str(
                metadata["canonical_patient_order_sha256"]
            ),
            "canonical_patient_label_sha256": str(
                metadata["canonical_patient_label_sha256"]
            ),
            "shared_initialization_sha256": str(
                checkpoint_contract.get(
                    "shared_initialization_sha256",
                    resolved.get("shared_initialization_sha256", ""),
                )
            ),
        }
        std = float(selection["selected_representation_std"])
        stability_rows.append(
            {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "model": model,
                "selected_epoch": int(selection["selected_epoch"]),
                "selection_mode": str(selection["selection_mode"]),
                "val_state_loss": float(selection["selected_validation_base_loss"]),
                "val_ftv_loss": float(selection["selected_validation_ftv_loss"]),
                "representation_std": std,
                "lambda_ftv": expected_lambda,
                "selected_scalars_finite": True,
                "checkpoint_tensors_finite": bool(tensors_finite),
                "checkpoint_tensor_count": int(tensor_count),
                "feature_finite": True,
                "no_collapse": bool(std >= 0.05),
                "architecture_contract_verified": bool(config.audit_checkpoints),
                "test_data_used": False,
                "checkpoint_sha256": checkpoint_sha,
                "feature_sha256": feature_sha,
                "shared_initialization_sha256": assets[(seed, model, fold)][
                    "shared_initialization_sha256"
                ],
            }
        )
        history_manifest.append(
            _manifest_row(
                "training_history",
                history_path,
                seed_base=seed,
                model=model,
                fold=fold,
                rows=len(history),
            )
        )
        selection_manifest.append(
            _manifest_row(
                "training_selection",
                selection_path,
                seed_base=seed,
                model=model,
                fold=fold,
            )
        )
        input_manifest.extend(
            (
                _manifest_row(
                    "checkpoint", checkpoint, seed_base=seed, model=model, fold=fold
                ),
                _manifest_row(
                    "resolved_run",
                    resolved_path,
                    seed_base=seed,
                    model=model,
                    fold=fold,
                ),
                _manifest_row(
                    "feature", feature_path, seed_base=seed, model=model, fold=fold
                ),
                _manifest_row(
                    "feature_metadata",
                    feature_metadata_path,
                    seed_base=seed,
                    model=model,
                    fold=fold,
                ),
                _manifest_row(
                    "feature_fragment",
                    feature_fragment_path,
                    seed_base=seed,
                    model=model,
                    fold=fold,
                    rows=808,
                ),
            )
        )
    # patient order/split/label 必须在 G1/G3 和五个 representation seeds 间完全相同。
    for fold in FOLDS:
        fingerprints = {
            feature_fingerprints[(seed, model, fold)]
            for seed in SEEDS
            for model in MODELS
        }
        if len(fingerprints) != 1:
            raise AnalysisInputError(
                f"feature canonical patient/split/label 跨 seed/model 漂移: fold={fold}"
            )
    if config.audit_checkpoints:
        for seed in SEEDS:
            for fold in FOLDS:
                g1, g3 = payloads[(seed, "G1", fold)], payloads[(seed, "G3", fold)]
                _compare_common_checkpoint_contract(g1, g3, f"seed={seed}/fold={fold}")
                g1_baseline = dict(g1.get("baseline_selection_contract", {}))
                if (
                    g1_baseline.get("paired_model") is not None
                    or g1_baseline.get("base_metric")
                    != "validation_normalized_next_state_loss_without_sigreg"
                    or g1_baseline.get("baseline_checkpoint") is not None
                    or g1_baseline.get("baseline_checkpoint_sha256") is not None
                    or g1_baseline.get("maximum_relative_degradation") is not None
                    or g1_baseline.get("allowed_val_base_loss") is not None
                ):
                    raise AnalysisInputError(
                        f"G1 baseline selection contract 漂移: seed={seed}/fold={fold}"
                    )
                baseline = dict(g3.get("baseline_selection_contract", {}))
                if (
                    baseline.get("paired_model") != "G1"
                    or baseline.get("base_metric")
                    != "validation_normalized_next_state_loss_without_sigreg"
                    or str(baseline.get("baseline_checkpoint_sha256"))
                    != assets[(seed, "G1", fold)]["checkpoint_sha256"]
                    or not math.isclose(
                        float(baseline.get("baseline_val_base_loss", math.nan)),
                        float(assets[(seed, "G1", fold)]["selected_val_base_loss"]),
                        rel_tol=0,
                        abs_tol=1e-10,
                    )
                    or not math.isclose(
                        float(baseline.get("maximum_relative_degradation", math.nan)),
                        0.05,
                        rel_tol=0,
                        abs_tol=0,
                    )
                    or not math.isclose(
                        float(baseline.get("allowed_val_base_loss", math.nan)),
                        1.05
                        * float(assets[(seed, "G1", fold)]["selected_val_base_loss"]),
                        rel_tol=0,
                        abs_tol=1e-10,
                    )
                ):
                    raise AnalysisInputError(
                        f"G3 paired baseline checkpoint SHA 未闭合: seed={seed}/fold={fold}"
                    )
        for fold in FOLDS:
            if (
                len(
                    {
                        assets[(seed, "G1", fold)]["shared_initialization_sha256"]
                        for seed in SEEDS
                    }
                )
                <= 1
            ):
                raise AnalysisInputError(
                    f"不同 seed 的 initialization 全相同: fold={fold}"
                )
    # downstream prediction provenance 必须闭合到刚审计的 feature/checkpoint。
    for frame, label in ((probe, "probe"), (pcr, "pCR")):
        for (seed, model, fold), part in frame.groupby(["seed_base", "model", "fold"]):
            expected = assets[(int(seed), str(model), int(fold))]
            if (
                set(part["source_feature_sha256"].astype(str))
                != {expected["feature_sha256"]}
                or set(part["source_checkpoint_sha256"].astype(str))
                != {expected["checkpoint_sha256"]}
                or set(part["feature_extractor_sha256"].astype(str))
                != {expected["feature_extractor_sha256"]}
                or set(part["canonical_patient_order_sha256"].astype(str))
                != {expected["canonical_patient_order_sha256"]}
                or set(part["canonical_patient_label_sha256"].astype(str))
                != {expected["canonical_patient_label_sha256"]}
                or set(part["fold_manifest_sha256"].astype(str))
                != {EXPECTED_FOLD_MANIFEST_SHA256}
            ):
                raise AnalysisInputError(
                    f"{label} source hash 未闭合: seed={seed}/{model}/fold={fold}"
                )
    downstream, downstream_manifest = _audit_downstream_selections(
        config, assets, probe, pcr
    )
    stability = pd.DataFrame(stability_rows)
    for seed in SEEDS:
        for fold in FOLDS:
            g1 = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G1")
            ].iloc[0]
            mask = (
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G3")
            )
            degradation = (
                float(stability.loc[mask, "val_state_loss"].iloc[0])
                - float(g1.val_state_loss)
            ) / max(float(g1.val_state_loss), 1e-12)
            stability.loc[mask, "paired_baseline"] = "G1"
            stability.loc[mask, "base_degradation_fraction"] = degradation
            stability.loc[mask, "base_pass"] = bool(
                math.isfinite(degradation) and degradation <= 0.05 + 1e-12
            )
    stability["base_pass"] = stability["base_pass"].astype("boolean")
    return (
        stability,
        pd.DataFrame(history_manifest),
        pd.DataFrame(selection_manifest),
        pd.concat(
            [pd.DataFrame(input_manifest), downstream_manifest], ignore_index=True
        ),
        downstream,
    )


def _corr(target: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    if (
        len(target) < 2
        or np.all(target == target[0])
        or np.all(prediction == prediction[0])
    ):
        return math.nan
    result = (
        spearmanr(target, prediction)
        if kind == "spearman"
        else pearsonr(target, prediction)
    )
    return float(result.statistic)


def regression_metrics(group: pd.DataFrame) -> dict[str, Any]:
    target = group["y_true"].to_numpy(dtype=float)
    prediction = group["y_pred"].to_numpy(dtype=float)
    baseline = group["b0_prediction"].to_numpy(dtype=float)
    rmse = float(math.sqrt(mean_squared_error(target, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(target, baseline)))
    target_variance = float(np.var(target, ddof=0))
    prediction_variance = float(np.var(prediction, ddof=0))
    return {
        "n": int(len(group)),
        "n_patients": int(group["patient_id"].nunique()),
        "n_folds": int(group["fold"].nunique()),
        "spearman": _corr(target, prediction, "spearman"),
        "pearson": _corr(target, prediction, "pearson"),
        "r2": float(r2_score(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": rmse,
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "target_variance": target_variance,
        "prediction_variance": prediction_variance,
        "prediction_target_variance_ratio": (
            prediction_variance / target_variance if target_variance > 0 else math.nan
        ),
    }


def classification_metrics(group: pd.DataFrame) -> dict[str, Any]:
    target = group["y_true"].to_numpy(dtype=int)
    probability = group["probability"].to_numpy(dtype=float)
    label = group["predicted_label"].to_numpy(dtype=int)
    if len(np.unique(target)) != 2:
        raise AnalysisInputError("pCR metric group 缺一个 class")
    return {
        "n": int(len(group)),
        "n_patients": int(group["patient_id"].nunique()),
        "n_folds": int(group["fold"].nunique()),
        "positive": int(target.sum()),
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "accuracy": float(accuracy_score(target, label)),
        "sensitivity": float(recall_score(target, label, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(target, label, pos_label=0, zero_division=0)),
    }


def _metric_rows(
    frame: pd.DataFrame, group_columns: Sequence[str], classifier: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric = classification_metrics if classifier else regression_metrics
    for keys, part in frame.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(metric(part))
        rows.append(row)
    output = pd.DataFrame(rows)
    numeric = (
        list(CLASSIFICATION_METRICS)
        if classifier
        else [
            metric
            for metric in REGRESSION_METRICS
            if metric not in {"spearman", "pearson"}
        ]
    )
    if (
        not output.empty
        and not np.isfinite(output[numeric].to_numpy(dtype=float)).all()
    ):
        raise AnalysisInputError(
            f"聚合必要 metric 含 nonfinite: groups={group_columns}"
        )
    return output


def _strict_mean(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)) if len(array) else math.nan


def _strict_sd(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if len(array) >= 2 else math.nan


def build_metric_tables(
    probe: pd.DataFrame, pcr: pd.DataFrame, stability: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    probe_seed = _metric_rows(probe, ("seed_base", "model", "task", "cell"))
    probe_fold = _metric_rows(probe, ("seed_base", "fold", "model", "task", "cell"))
    pcr_seed_model = _metric_rows(
        pcr, ("seed_base", "model", "decision_point"), classifier=True
    )

    paired_pcr_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for point in DECISION_POINTS:
            subset = pcr_seed_model.loc[
                pcr_seed_model["seed_base"].eq(seed)
                & pcr_seed_model["decision_point"].eq(point)
            ].set_index("model")
            row: dict[str, Any] = {"seed_base": seed, "decision_point": point}
            for metric in CLASSIFICATION_METRICS:
                row[f"g1_{metric}"] = float(subset.loc["G1", metric])
                row[f"g3_{metric}"] = float(subset.loc["G3", metric])
                row[f"delta_{metric}"] = row[f"g3_{metric}"] - row[f"g1_{metric}"]
            row["n_patients"] = int(subset.loc["G1", "n_patients"])
            paired_pcr_rows.append(row)
    pcr_secondary = pd.DataFrame(paired_pcr_rows)

    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        cells = probe_seed.loc[probe_seed["seed_base"].eq(seed)]
        fold_cells = probe_fold.loc[probe_fold["seed_base"].eq(seed)]
        static = cells.loc[cells["task"].eq("static")]
        dynamic = cells.loc[cells["task"].eq("change")]
        d_s = float(
            _strict_mean(static.loc[static["model"].eq("G3"), "spearman"])
            - _strict_mean(static.loc[static["model"].eq("G1"), "spearman"])
        )
        d_d = float(
            _strict_mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"])
            - _strict_mean(dynamic.loc[dynamic["model"].eq("G1"), "spearman"])
        )
        d_d_r2 = float(
            _strict_mean(dynamic.loc[dynamic["model"].eq("G3"), "r2"])
            - _strict_mean(dynamic.loc[dynamic["model"].eq("G1"), "r2"])
        )
        longitudinal = pcr_secondary.loc[
            pcr_secondary["seed_base"].eq(seed)
            & pcr_secondary["decision_point"].isin(("T0-T1", "T0-T2"))
        ]
        failures = stability.loc[
            stability["seed_base"].eq(seed)
            & stability["model"].eq("G3")
            & stability["base_pass"].eq(False)
        ]
        seed_row: dict[str, Any] = {
            "seed_base": seed,
            "dS": d_s,
            "static_delta_spearman": d_s,
            "dD": d_d,
            "delta_ftv_delta_spearman": d_d,
            "dD_R2": d_d_r2,
            "delta_ftv_delta_r2": d_d_r2,
            "pcr_longitudinal_delta_auroc": _strict_mean(longitudinal["delta_auroc"]),
            "pcr_longitudinal_delta_auprc": _strict_mean(longitudinal["delta_auprc"]),
            "failed_fold_count": int(len(failures)),
            "static_positive": bool(d_s > 0),
            "dynamic_positive": bool(d_d > 0),
            "pooled_oof_probe_patients": int(
                fold_cells.loc[fold_cells["model"].eq("G1"), "n_patients"].sum() / 7
            ),
        }
        for point in DECISION_POINTS:
            point_row = pcr_secondary.loc[
                pcr_secondary["seed_base"].eq(seed)
                & pcr_secondary["decision_point"].eq(point)
            ].iloc[0]
            token = point.replace("-", "_")
            for metric in ("auroc", "auprc"):
                seed_row[f"pcr_{token}_delta_{metric}"] = float(
                    point_row[f"delta_{metric}"]
                )
        seed_rows.append(seed_row)
    seed_level = pd.DataFrame(seed_rows)

    seed_fold_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            cells = probe_fold.loc[
                probe_fold["seed_base"].eq(seed) & probe_fold["fold"].eq(fold)
            ]
            static = cells.loc[cells["task"].eq("static")]
            dynamic = cells.loc[cells["task"].eq("change")]
            d_s_sf = float(
                _strict_mean(static.loc[static["model"].eq("G3"), "spearman"])
                - _strict_mean(static.loc[static["model"].eq("G1"), "spearman"])
            )
            d_d_sf = float(
                _strict_mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"])
                - _strict_mean(dynamic.loc[dynamic["model"].eq("G1"), "spearman"])
            )
            g1 = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G1")
            ].iloc[0]
            g3 = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G3")
            ].iloc[0]
            seed_fold_rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "dS_sf": d_s_sf,
                    "dD_sf": d_d_sf,
                    "D": float(g3.base_degradation_fraction),
                    "base_pass": bool(g3.base_pass),
                    "g1_val_state_loss": float(g1.val_state_loss),
                    "g3_val_state_loss": float(g3.val_state_loss),
                    "g1_selected_epoch": int(g1.selected_epoch),
                    "g3_selected_epoch": int(g3.selected_epoch),
                    "g1_representation_std": float(g1.representation_std),
                    "g3_representation_std": float(g3.representation_std),
                }
            )
    seed_fold = pd.DataFrame(seed_fold_rows)

    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        part = seed_fold.loc[seed_fold["fold"].eq(fold)]
        fold_rows.append(
            {
                "fold": fold,
                "static_delta_spearman_mean": _strict_mean(part["dS_sf"]),
                "static_delta_spearman_sd": _strict_sd(part["dS_sf"]),
                "delta_ftv_delta_spearman_mean": _strict_mean(part["dD_sf"]),
                "delta_ftv_delta_spearman_sd": _strict_sd(part["dD_sf"]),
                "base_failure_count": int((~part["base_pass"]).sum()),
                "base_failure_rate": float((~part["base_pass"]).mean()),
                "base_degradation_mean": _strict_mean(part["D"]),
                "base_degradation_sd": _strict_sd(part["D"]),
            }
        )
    return {
        "probe_seed_cell_metrics": probe_seed,
        "probe_seed_fold_cell_metrics": probe_fold,
        "pcr_seed_model_metrics": pcr_seed_model,
        "pcr_secondary_seed_metrics": pcr_secondary,
        "seed_level_robustness": seed_level,
        "seed_fold_effects": seed_fold,
        "fold_level_robustness": pd.DataFrame(fold_rows),
    }


def _bootstrap_indices(
    base: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """在 outer fold 内分层重采患者；返回共享的 row index matrix。"""

    if replicates <= 0:
        raise ValueError("bootstrap replicates 必须为正")
    parts: list[np.ndarray] = []
    for fold in FOLDS:
        positions = np.flatnonzero(base["fold"].to_numpy(dtype=int) == fold)
        if len(positions) == 0:
            raise AnalysisInputError(f"bootstrap 缺 fold={fold}")
        parts.append(
            rng.choice(positions, size=(replicates, len(positions)), replace=True)
        )
    return np.concatenate(parts, axis=1)


def _rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_center = x - x.mean(axis=1, keepdims=True)
    y_center = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(x_center * x_center, axis=1) * np.sum(y_center * y_center, axis=1)
    )
    numerator = np.sum(x_center * y_center, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(x), np.nan, dtype=float),
        where=denominator > 0,
    )


def _bootstrap_spearman(
    target: np.ndarray, prediction: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    target_sample = target[indices]
    prediction_sample = prediction[indices]
    return _rowwise_corr(
        rankdata(target_sample, axis=1), rankdata(prediction_sample, axis=1)
    )


def _bootstrap_r2(
    target: np.ndarray, prediction: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    target_sample = target[indices]
    prediction_sample = prediction[indices]
    denominator = np.sum(
        (target_sample - target_sample.mean(axis=1, keepdims=True)) ** 2, axis=1
    )
    numerator = np.sum((target_sample - prediction_sample) ** 2, axis=1)
    return np.divide(
        denominator - numerator,
        denominator,
        out=np.full(len(indices), np.nan, dtype=float),
        where=denominator > 0,
    )


def _aligned_probe_arrays(
    probe: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    seed_part = probe.loc[probe["seed_base"].eq(seed)]
    base = (
        seed_part.loc[
            seed_part["model"].eq("G1")
            & seed_part["task"].eq("static")
            & seed_part["cell"].eq("T0"),
            ["patient_id", "fold"],
        ]
        .sort_values(["fold", "patient_id"])
        .reset_index(drop=True)
    )
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for task, cells in (("static", TIMEPOINTS), ("change", TRANSITIONS)):
        for cell in cells:
            subset = seed_part.loc[
                seed_part["task"].eq(task) & seed_part["cell"].eq(cell)
            ]
            by_model: dict[str, pd.DataFrame] = {}
            for model in MODELS:
                part = (
                    subset.loc[subset["model"].eq(model)]
                    .sort_values(["fold", "patient_id"])
                    .reset_index(drop=True)
                )
                if not part[["patient_id", "fold"]].equals(base):
                    raise AnalysisInputError(
                        f"probe bootstrap patient order 不闭合: seed={seed}/{task}/{cell}/{model}"
                    )
                by_model[model] = part
            if not np.array_equal(by_model["G1"]["y_true"], by_model["G3"]["y_true"]):
                raise AnalysisInputError(
                    f"probe paired target 不一致: seed={seed}/{task}/{cell}"
                )
            arrays[(task, cell)] = {
                "target": by_model["G1"]["y_true"].to_numpy(dtype=float),
                "G1": by_model["G1"]["y_pred"].to_numpy(dtype=float),
                "G3": by_model["G3"]["y_pred"].to_numpy(dtype=float),
            }
    return base, arrays


def _probe_endpoint_bootstrap(
    arrays: Mapping[tuple[str, str], Mapping[str, np.ndarray]], indices: np.ndarray
) -> dict[str, np.ndarray]:
    static_s: list[np.ndarray] = []
    dynamic_s: list[np.ndarray] = []
    dynamic_r2: list[np.ndarray] = []
    for task, cells in (("static", TIMEPOINTS), ("change", TRANSITIONS)):
        for cell in cells:
            item = arrays[(task, cell)]
            target = item["target"]
            difference_s = _bootstrap_spearman(
                target, item["G3"], indices
            ) - _bootstrap_spearman(target, item["G1"], indices)
            if task == "static":
                static_s.append(difference_s)
            else:
                dynamic_s.append(difference_s)
                dynamic_r2.append(
                    _bootstrap_r2(target, item["G3"], indices)
                    - _bootstrap_r2(target, item["G1"], indices)
                )
    return {
        "dS": np.mean(np.stack(static_s), axis=0),
        "dD": np.mean(np.stack(dynamic_s), axis=0),
        "dD_R2": np.mean(np.stack(dynamic_r2), axis=0),
    }


def _aligned_pcr_arrays(
    pcr: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    seed_part = pcr.loc[pcr["seed_base"].eq(seed)]
    base = (
        seed_part.loc[
            seed_part["model"].eq("G1") & seed_part["decision_point"].eq("T0"),
            ["patient_id", "fold"],
        ]
        .sort_values(["fold", "patient_id"])
        .reset_index(drop=True)
    )
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for point in DECISION_POINTS:
        subset = seed_part.loc[seed_part["decision_point"].eq(point)]
        by_model: dict[str, pd.DataFrame] = {}
        for model in MODELS:
            part = (
                subset.loc[subset["model"].eq(model)]
                .sort_values(["fold", "patient_id"])
                .reset_index(drop=True)
            )
            if not part[["patient_id", "fold"]].equals(base):
                raise AnalysisInputError(
                    f"pCR bootstrap patient order 不闭合: seed={seed}/{point}/{model}"
                )
            by_model[model] = part
        if not np.array_equal(by_model["G1"]["y_true"], by_model["G3"]["y_true"]):
            raise AnalysisInputError(f"pCR paired target 不一致: seed={seed}/{point}")
        arrays[point] = {
            "target": by_model["G1"]["y_true"].to_numpy(dtype=int),
            "G1": by_model["G1"]["probability"].to_numpy(dtype=float),
            "G3": by_model["G3"]["probability"].to_numpy(dtype=float),
        }
    return base, arrays


def _bootstrap_auroc(
    target: np.ndarray, probability: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    labels = target[indices]
    scores = probability[indices]
    positives = labels.sum(axis=1)
    negatives = labels.shape[1] - positives
    ranks = rankdata(scores, axis=1)
    numerator = np.sum(ranks * labels, axis=1) - positives * (positives + 1) / 2.0
    denominator = positives * negatives
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(indices), np.nan, dtype=float),
        where=denominator > 0,
    )


def _bootstrap_auprc(
    target: np.ndarray, probability: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    output = np.full(len(indices), np.nan, dtype=float)
    for index, row in enumerate(indices):
        labels = target[row]
        if len(np.unique(labels)) == 2:
            output[index] = average_precision_score(labels, probability[row])
    return output


def _pcr_endpoint_bootstrap(
    arrays: Mapping[str, Mapping[str, np.ndarray]], indices: np.ndarray
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    longitudinal_auroc: list[np.ndarray] = []
    longitudinal_auprc: list[np.ndarray] = []
    for point in DECISION_POINTS:
        item = arrays[point]
        delta_auroc = _bootstrap_auroc(
            item["target"], item["G3"], indices
        ) - _bootstrap_auroc(item["target"], item["G1"], indices)
        delta_auprc = _bootstrap_auprc(
            item["target"], item["G3"], indices
        ) - _bootstrap_auprc(item["target"], item["G1"], indices)
        token = point.replace("-", "_")
        output[f"pcr_{token}_delta_auroc"] = delta_auroc
        output[f"pcr_{token}_delta_auprc"] = delta_auprc
        if point != "T0":
            longitudinal_auroc.append(delta_auroc)
            longitudinal_auprc.append(delta_auprc)
    output["pcr_longitudinal_delta_auroc"] = np.mean(
        np.stack(longitudinal_auroc), axis=0
    )
    output["pcr_longitudinal_delta_auprc"] = np.mean(
        np.stack(longitudinal_auprc), axis=0
    )
    return output


def _quantile_ci(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan, math.nan, 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def conditional_bootstrap(
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    seed_level: pd.DataFrame,
    pcr_secondary: pd.DataFrame,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        probe_base, probe_arrays = _aligned_probe_arrays(probe, seed)
        probe_rng = np.random.default_rng(
            _stable_seed(random_seed, "conditional", seed, "probe")
        )
        probe_indices = _bootstrap_indices(probe_base, replicates, probe_rng)
        probe_boot = _probe_endpoint_bootstrap(probe_arrays, probe_indices)
        seed_row = seed_level.loc[seed_level["seed_base"].eq(seed)].iloc[0]
        for endpoint, values in probe_boot.items():
            low, high, finite = _quantile_ci(values)
            rows.append(
                {
                    "seed_base": seed,
                    "cohort": "FTV",
                    "endpoint": endpoint,
                    "estimate": float(seed_row[endpoint]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": replicates,
                    "finite_replicates": finite,
                    "rng_seed": random_seed,
                    "bootstrap_unit": "patient_within_outer_fold_same_draw_across_G1_G3_and_cells",
                }
            )
        pcr_base, pcr_arrays = _aligned_pcr_arrays(pcr, seed)
        pcr_rng = np.random.default_rng(
            _stable_seed(random_seed, "conditional", seed, "pcr")
        )
        pcr_indices = _bootstrap_indices(pcr_base, replicates, pcr_rng)
        pcr_boot = _pcr_endpoint_bootstrap(pcr_arrays, pcr_indices)
        point_table = pcr_secondary.loc[pcr_secondary["seed_base"].eq(seed)].set_index(
            "decision_point"
        )
        for endpoint, values in pcr_boot.items():
            if endpoint.startswith("pcr_longitudinal"):
                estimate = float(seed_row[endpoint])
            else:
                metric = "auroc" if endpoint.endswith("auroc") else "auprc"
                point_token = endpoint.removeprefix("pcr_").removesuffix(
                    f"_delta_{metric}"
                )
                point = point_token.replace("_", "-")
                estimate = float(point_table.loc[point, f"delta_{metric}"])
            low, high, finite = _quantile_ci(values)
            rows.append(
                {
                    "seed_base": seed,
                    "cohort": "pCR",
                    "endpoint": endpoint,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": replicates,
                    "finite_replicates": finite,
                    "rng_seed": random_seed,
                    "bootstrap_unit": "patient_within_outer_fold_same_draw_across_G1_G3_and_decision_points",
                }
            )
    return pd.DataFrame(rows)


def crossed_bootstrap(
    probe: pd.DataFrame,
    seed_level: pd.DataFrame,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """支持性 crossed bootstrap：同步抽患者，并重采 training seeds。"""

    base, first_arrays = _aligned_probe_arrays(probe, SEEDS[0])
    rng = np.random.default_rng(random_seed)
    indices = _bootstrap_indices(base, replicates, rng)
    endpoint_matrix: dict[str, np.ndarray] = {
        endpoint: np.empty((replicates, len(SEEDS)), dtype=float)
        for endpoint in ("dS", "dD", "dD_R2")
    }
    for column, seed in enumerate(SEEDS):
        if seed == SEEDS[0]:
            arrays = first_arrays
        else:
            current_base, arrays = _aligned_probe_arrays(probe, seed)
            if not current_base.equals(base):
                raise AnalysisInputError("crossed bootstrap patient order 跨 seed 漂移")
        values = _probe_endpoint_bootstrap(arrays, indices)
        for endpoint in endpoint_matrix:
            endpoint_matrix[endpoint][:, column] = values[endpoint]
    sampled_seeds = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)))
    replicate_rows = np.arange(replicates)[:, None]
    output: list[dict[str, Any]] = []
    for endpoint, matrix in endpoint_matrix.items():
        crossed = matrix[replicate_rows, sampled_seeds].mean(axis=1)
        low, high, finite = _quantile_ci(crossed)
        output.append(
            {
                "endpoint": endpoint,
                "estimate": _strict_mean(seed_level[endpoint]),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": replicates,
                "finite_replicates": finite,
                "rng_seed": random_seed,
                "bootstrap_unit": "resampled_training_seeds_and_synchronized_patients_within_outer_fold",
                "used_for_decision": False,
            }
        )
    return pd.DataFrame(output)


def _seed_effect_from_raw(part: pd.DataFrame) -> dict[str, float]:
    metrics = _metric_rows(part, ("model", "task", "cell"))
    static = metrics.loc[metrics["task"].eq("static")]
    dynamic = metrics.loc[metrics["task"].eq("change")]
    return {
        "dS": float(
            _strict_mean(static.loc[static["model"].eq("G3"), "spearman"])
            - _strict_mean(static.loc[static["model"].eq("G1"), "spearman"])
        ),
        "dD": float(
            _strict_mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"])
            - _strict_mean(dynamic.loc[dynamic["model"].eq("G1"), "spearman"])
        ),
        "dD_R2": float(
            _strict_mean(dynamic.loc[dynamic["model"].eq("G3"), "r2"])
            - _strict_mean(dynamic.loc[dynamic["model"].eq("G1"), "r2"])
        ),
    }


def leave_one_out_sensitivity(
    probe: pd.DataFrame, seed_level: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted_seed in SEEDS:
        kept = seed_level.loc[~seed_level["seed_base"].eq(omitted_seed)]
        rows.append(
            {
                "scope": "leave_one_seed_out",
                "omitted_seed": omitted_seed,
                "omitted_fold": pd.NA,
                "seed_base": pd.NA,
                "static_mean": _strict_mean(kept["dS"]),
                "dynamic_mean": _strict_mean(kept["dD"]),
                "dynamic_r2_mean": _strict_mean(kept["dD_R2"]),
                "spearman_recomputed_from_patient_rows": False,
                "n_training_seeds": len(kept),
            }
        )
    for omitted_fold in FOLDS:
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            # 关键：删除患者行后从剩余四折 pooled OOF 重新计算 Spearman/R²。
            subset = probe.loc[
                probe["seed_base"].eq(seed) & ~probe["fold"].eq(omitted_fold)
            ]
            effect = _seed_effect_from_raw(subset)
            per_seed.append({"seed_base": seed, **effect})
            rows.append(
                {
                    "scope": "leave_one_fold_out_seed",
                    "omitted_seed": pd.NA,
                    "omitted_fold": omitted_fold,
                    "seed_base": seed,
                    "static_mean": effect["dS"],
                    "dynamic_mean": effect["dD"],
                    "dynamic_r2_mean": effect["dD_R2"],
                    "spearman_recomputed_from_patient_rows": True,
                    "n_training_seeds": 1,
                }
            )
        per_seed_frame = pd.DataFrame(per_seed)
        rows.append(
            {
                "scope": "leave_one_fold_out_across_seed",
                "omitted_seed": pd.NA,
                "omitted_fold": omitted_fold,
                "seed_base": pd.NA,
                "static_mean": _strict_mean(per_seed_frame["dS"]),
                "dynamic_mean": _strict_mean(per_seed_frame["dD"]),
                "dynamic_r2_mean": _strict_mean(per_seed_frame["dD_R2"]),
                "spearman_recomputed_from_patient_rows": True,
                "n_training_seeds": len(SEEDS),
            }
        )
    return pd.DataFrame(rows)


def seed_uncertainty(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_columns = {
        "dS": "primary_static_spearman",
        "dD": "primary_dynamic_spearman",
        "dD_R2": "descriptive_dynamic_r2",
        "pcr_longitudinal_delta_auroc": "secondary_pcr_longitudinal_auroc",
        "pcr_longitudinal_delta_auprc": "secondary_pcr_longitudinal_auprc",
    }
    for point in DECISION_POINTS:
        token = point.replace("-", "_")
        for metric in ("auroc", "auprc"):
            endpoint_columns[f"pcr_{token}_delta_{metric}"] = (
                f"secondary_pcr_{token}_{metric}"
            )
    n = len(seed_level)
    critical = float(student_t.ppf(0.975, n - 1))
    for column, endpoint in endpoint_columns.items():
        values = seed_level[column].to_numpy(dtype=float)
        if len(values) != len(SEEDS):
            raise AnalysisInputError(f"seed endpoint 行数不可验证: {column}")
        verifiable = bool(np.isfinite(values).all())
        finite = values[np.isfinite(values)]
        positive_n = int(np.count_nonzero(finite > 0))
        if not verifiable:
            rows.append(
                {
                    "endpoint": endpoint,
                    "column": column,
                    "n_seeds": n,
                    "finite_seeds": int(len(finite)),
                    "verifiable": False,
                    "mean": math.nan,
                    "sample_sd": math.nan,
                    "minimum": math.nan,
                    "median": math.nan,
                    "maximum": math.nan,
                    "positive_n": positive_n,
                    "positive_rate": math.nan,
                    "t_critical": critical,
                    "t_ci_low": math.nan,
                    "t_ci_high": math.nan,
                    "exact_sign_test_p_two_sided": math.nan,
                    "clopper_pearson_positive_rate_low": math.nan,
                    "clopper_pearson_positive_rate_high": math.nan,
                    "used_for_R1": column == "dS",
                    "used_for_R2": column == "dD",
                }
            )
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        margin = critical * sd / math.sqrt(n)
        cp_low = (
            0.0
            if positive_n == 0
            else float(beta.ppf(0.025, positive_n, n - positive_n + 1))
        )
        cp_high = (
            1.0
            if positive_n == n
            else float(beta.ppf(0.975, positive_n + 1, n - positive_n))
        )
        rows.append(
            {
                "endpoint": endpoint,
                "column": column,
                "n_seeds": n,
                "finite_seeds": n,
                "verifiable": True,
                "mean": mean,
                "sample_sd": sd,
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
                "positive_n": positive_n,
                "positive_rate": positive_n / n,
                "t_critical": critical,
                "t_ci_low": mean - margin,
                "t_ci_high": mean + margin,
                "exact_sign_test_p_two_sided": float(
                    binomtest(positive_n, n, 0.5).pvalue
                ),
                "clopper_pearson_positive_rate_low": cp_low,
                "clopper_pearson_positive_rate_high": cp_high,
                "used_for_R1": column == "dS",
                "used_for_R2": column == "dD",
            }
        )
    return pd.DataFrame(rows)


def variance_decomposition(
    grid: pd.DataFrame, value: str, endpoint: str, primary: bool
) -> dict[str, Any]:
    matrix = (
        grid.pivot(index="seed_base", columns="fold", values=value)
        .reindex(index=SEEDS, columns=FOLDS)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (5, 5):
        raise AnalysisInputError(
            f"variance decomposition 需要 balanced 5x5 grid: {value}"
        )
    if not np.isfinite(matrix).all():
        return {
            "endpoint": endpoint,
            "value_column": value,
            "primary": bool(primary),
            "verifiable": False,
            "grand_mean": math.nan,
            "ss_seed": math.nan,
            "ss_fold": math.nan,
            "ss_residual": math.nan,
            "df_seed": 4,
            "df_fold": 4,
            "df_residual": 16,
            "ms_seed": math.nan,
            "ms_fold": math.nan,
            "ms_residual": math.nan,
            "raw_seed_component": math.nan,
            "raw_fold_component": math.nan,
            "raw_interaction_sampling_component": math.nan,
            "clipped_seed_component": math.nan,
            "clipped_fold_component": math.nan,
            "clipped_interaction_sampling_component": math.nan,
            "sqrt_seed_component": math.nan,
            "sqrt_fold_component": math.nan,
            "sqrt_interaction_sampling_component": math.nan,
            "seed_share": 0.0,
            "fold_share": 0.0,
            "interaction_sampling_share": 0.0,
            "dominance": "unverifiable",
            "residual_label": "seed×fold interaction + metric sampling error",
            "no_cell_replication": True,
        }
    grand = float(matrix.mean())
    rows = matrix.mean(axis=1)
    columns = matrix.mean(axis=0)
    ss_seed = float(5 * np.sum((rows - grand) ** 2))
    ss_fold = float(5 * np.sum((columns - grand) ** 2))
    residual = matrix - rows[:, None] - columns[None, :] + grand
    ss_residual = float(np.sum(residual**2))
    ms_seed, ms_fold, ms_residual = ss_seed / 4, ss_fold / 4, ss_residual / 16
    raw = np.asarray(
        [(ms_seed - ms_residual) / 5, (ms_fold - ms_residual) / 5, ms_residual],
        dtype=float,
    )
    clipped = np.maximum(raw, 0.0)
    total = float(clipped.sum())
    shares = np.zeros(3, dtype=float) if total == 0 else clipped / total
    labels = ("seed", "fold", "interaction_and_metric_sampling_error")
    if total == 0:
        dominance = "no_variation"
    elif float(shares.max()) > 0.5:
        dominance = labels[int(np.argmax(shares))]
    else:
        dominance = "mixed"
    return {
        "endpoint": endpoint,
        "value_column": value,
        "primary": bool(primary),
        "verifiable": True,
        "grand_mean": grand,
        "ss_seed": ss_seed,
        "ss_fold": ss_fold,
        "ss_residual": ss_residual,
        "df_seed": 4,
        "df_fold": 4,
        "df_residual": 16,
        "ms_seed": ms_seed,
        "ms_fold": ms_fold,
        "ms_residual": ms_residual,
        "raw_seed_component": float(raw[0]),
        "raw_fold_component": float(raw[1]),
        "raw_interaction_sampling_component": float(raw[2]),
        "clipped_seed_component": float(clipped[0]),
        "clipped_fold_component": float(clipped[1]),
        "clipped_interaction_sampling_component": float(clipped[2]),
        "sqrt_seed_component": float(math.sqrt(clipped[0])),
        "sqrt_fold_component": float(math.sqrt(clipped[1])),
        "sqrt_interaction_sampling_component": float(math.sqrt(clipped[2])),
        "seed_share": float(shares[0]),
        "fold_share": float(shares[1]),
        "interaction_sampling_share": float(shares[2]),
        "dominance": dominance,
        "residual_label": "seed×fold interaction + metric sampling error",
        "no_cell_replication": True,
    }


def variance_decomposition_table(seed_fold: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            variance_decomposition(seed_fold, "dD_sf", "dynamic_spearman", True),
            variance_decomposition(seed_fold, "dS_sf", "static_spearman", False),
            variance_decomposition(seed_fold, "D", "base_degradation", False),
        ]
    )


def variance_decomposition_self_test() -> dict[str, bool]:
    grid = pd.DataFrame(
        [(s, f) for s in SEEDS for f in FOLDS], columns=["seed_base", "fold"]
    )
    grid["constant"] = 0.25
    grid["seed_only"] = (
        grid["seed_base"]
        .map({seed: index for index, seed in enumerate(SEEDS)})
        .astype(float)
    )
    grid["fold_only"] = grid["fold"].astype(float)
    constant = variance_decomposition(grid, "constant", "constant", True)
    seed_only = variance_decomposition(grid, "seed_only", "seed_only", True)
    fold_only = variance_decomposition(grid, "fold_only", "fold_only", True)
    checks = {
        "constant_no_variation": constant["dominance"] == "no_variation"
        and constant["seed_share"]
        == constant["fold_share"]
        == constant["interaction_sampling_share"]
        == 0,
        "seed_only_dominates": seed_only["dominance"] == "seed"
        and seed_only["seed_share"] > 0.5,
        "fold_only_dominates": fold_only["dominance"] == "fold"
        and fold_only["fold_share"] > 0.5,
    }
    if not all(checks.values()):
        raise AssertionError(f"variance decomposition synthetic tests failed: {checks}")
    return checks


def compute_decision(
    seed_level: pd.DataFrame,
    seed_fold: pd.DataFrame,
    uncertainty: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    stability: pd.DataFrame,
    variance: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    def uncertainty_row(column: str) -> pd.Series:
        part = uncertainty.loc[uncertainty["column"].eq(column)]
        if len(part) != 1:
            raise AnalysisInputError(f"缺唯一 seed uncertainty row: {column}")
        return part.iloc[0]

    static_u, dynamic_u = uncertainty_row("dS"), uncertainty_row("dD")
    loso = leave_one_out.loc[leave_one_out["scope"].eq("leave_one_seed_out")]
    lofo = leave_one_out.loc[
        leave_one_out["scope"].eq("leave_one_fold_out_across_seed")
    ]
    r1_checks = {
        "all_five_dS_positive": bool(seed_level["dS"].gt(0).all()),
        "mean_dS_at_least_0_05": bool(float(static_u["mean"]) >= 0.05),
        "seed_t_ci_lower_positive": bool(float(static_u["t_ci_low"]) > 0),
        "all_leave_one_seed_out_means_positive": bool(loso["static_mean"].gt(0).all()),
        "all_leave_one_fold_out_recomputed_means_positive": bool(
            lofo["static_mean"].gt(0).all()
        ),
    }
    r2_checks = {
        "at_least_four_of_five_dD_positive": bool(seed_level["dD"].gt(0).sum() >= 4),
        "mean_dD_at_least_0_05": bool(float(dynamic_u["mean"]) >= 0.05),
        "seed_t_ci_lower_positive": bool(float(dynamic_u["t_ci_low"]) > 0),
        "all_leave_one_seed_out_means_positive": bool(loso["dynamic_mean"].gt(0).all()),
        "all_leave_one_fold_out_recomputed_means_positive": bool(
            lofo["dynamic_mean"].gt(0).all()
        ),
    }
    base = seed_fold["base_pass"].astype(bool)
    fold_failures = seed_fold.assign(failed=~base).groupby("fold")["failed"].sum()
    r3_checks = {
        "at_least_23_of_25_base_pass": bool(base.sum() >= 23),
        "each_fold_at_most_two_failures": bool(fold_failures.le(2).all()),
    }
    r4_checks = {
        "all_50_selected_scalars_finite": bool(
            len(stability) == 50
            and stability["selected_scalars_finite"].astype(bool).all()
        ),
        "all_50_checkpoint_tensors_finite": bool(
            len(stability) == 50
            and stability["checkpoint_tensors_finite"].astype(bool).all()
        ),
        "all_50_representation_std_at_least_0_05": bool(
            len(stability) == 50 and stability["representation_std"].ge(0.05).all()
        ),
        "all_50_feature_arrays_finite": bool(
            len(stability) == 50 and stability["feature_finite"].astype(bool).all()
        ),
    }
    gates = {
        "R1_static_reproducibility": all(r1_checks.values()),
        "R2_dynamic_reproducibility": all(r2_checks.values()),
        "R3_optimization_safety": all(r3_checks.values()),
        "R4_no_collapse": all(r4_checks.values()),
    }
    no_single = bool(
        loso[["static_mean", "dynamic_mean"]].gt(0).all().all()
        and lofo[["static_mean", "dynamic_mean"]].gt(0).all().all()
    )
    majority_failure = bool(fold_failures.ge(3).any())
    if all(gates.values()):
        conclusion = "ROBUST"
    elif (
        gates["R1_static_reproducibility"]
        and gates["R2_dynamic_reproducibility"]
        and no_single
        and not majority_failure
    ):
        conclusion = "PROMISING BUT UNSTABLE"
    else:
        conclusion = "NOT ROBUST"
    fold3_failures = int(fold_failures.loc[3])
    fold3_interpretation = (
        "不复现"
        if fold3_failures == 0
        else (
            "孤立/seed-dependent"
            if fold3_failures == 1
            else (
                "少数重复、提示冲突但不足以称系统"
                if fold3_failures == 2
                else "fold 3 systematic conflict"
            )
        )
    )
    failed_folds = int((fold_failures > 0).sum())
    general_instability = bool(failed_folds >= 2 and not majority_failure)
    dynamic_variance = variance.loc[variance["endpoint"].eq("dynamic_spearman")].iloc[0]
    decision = {
        "schema_version": SCHEMA_VERSION,
        "conclusion": conclusion,
        "gates": gates,
        "gate_details": {
            "R1": r1_checks,
            "R2": r2_checks,
            "R3": r3_checks,
            "R4": r4_checks,
        },
        "no_single_seed_or_fold_drive": no_single,
        "fold_majority_base_failure": majority_failure,
        "base_pass_count": int(base.sum()),
        "base_failure_count": int((~base).sum()),
        "base_failures_by_fold": {
            str(index): int(value) for index, value in fold_failures.items()
        },
        "fold3_failures": fold3_failures,
        "fold3_interpretation_cn": fold3_interpretation,
        "old_fold3_reference_degradation": 0.095934,
        "general_seed_fold_optimization_instability": general_instability,
        "dynamic_variance_dominance": str(dynamic_variance["dominance"]),
        "pcr_used_in_decision": False,
        "seed_t_ci_is_formal_uncertainty_gate": True,
        "conditional_patient_bootstrap_used_for_decision": False,
        "crossed_bootstrap_used_for_decision": False,
        "thresholds": {
            "R1_mean_dS": 0.05,
            "R2_mean_dD": 0.05,
            "R3_minimum_base_passes": 23,
            "maximum_failures_per_fold": 2,
            "maximum_base_degradation": 0.05,
            "minimum_representation_std": 0.05,
        },
    }
    gate_rows: list[dict[str, Any]] = []
    for gate, checks in (
        ("R1", r1_checks),
        ("R2", r2_checks),
        ("R3", r3_checks),
        ("R4", r4_checks),
    ):
        gate_rows.extend(
            {"gate": gate, "criterion": criterion, "passed": bool(passed)}
            for criterion, passed in checks.items()
        )
        gate_rows.append(
            {
                "gate": gate,
                "criterion": "overall",
                "passed": gates[
                    f"{gate}_"
                    + {
                        "R1": "static_reproducibility",
                        "R2": "dynamic_reproducibility",
                        "R3": "optimization_safety",
                        "R4": "no_collapse",
                    }[gate]
                ],
            }
        )
    return decision, pd.DataFrame(gate_rows)


def _setup_matplotlib() -> None:
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    matplotlib.rcParams["font.family"] = next(
        (name for name in candidates if name in available), "DejaVu Sans"
    )
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 130


def _seed_ci_figure(conditional: pd.DataFrame, endpoint: str, title: str) -> Figure:
    data = conditional.loc[
        conditional["cohort"].eq("FTV") & conditional["endpoint"].eq(endpoint)
    ].sort_values("seed_base")
    figure, axis = plt.subplots(figsize=(9, 5))
    estimate = data["estimate"].to_numpy(dtype=float)
    finite = (
        np.isfinite(estimate)
        & np.isfinite(data["ci_low"])
        & np.isfinite(data["ci_high"])
    )
    if finite.any():
        axis.errorbar(
            np.arange(len(data))[finite],
            estimate[finite],
            yerr=np.vstack(
                [
                    estimate[finite] - data.loc[finite, "ci_low"].to_numpy(),
                    data.loc[finite, "ci_high"].to_numpy() - estimate[finite],
                ]
            ),
            fmt="o",
            capsize=4,
            color="#2166ac",
        )
    else:
        axis.text(0.5, 0.5, "端点不可验证", transform=axis.transAxes, ha="center")
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.axhline(0.05, color="#b2182b", linestyle="--", linewidth=0.9)
    axis.set_xticks(np.arange(len(data)), data["seed_base"].astype(str))
    axis.set_xlabel("训练 seed_base")
    axis.set_ylabel("G3−G1 ΔSpearman")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def _heatmap(seed_fold: pd.DataFrame, value: str, title: str, label: str) -> Figure:
    matrix = (
        seed_fold.pivot(index="seed_base", columns="fold", values=value)
        .reindex(index=SEEDS, columns=FOLDS)
        .to_numpy(dtype=float)
    )
    finite = np.isfinite(matrix)
    limit = max(float(np.max(np.abs(matrix[finite]))), 1e-6) if finite.any() else 1.0
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row in range(5):
        for column in range(5):
            value_text = (
                f"{matrix[row, column]:+.3f}"
                if np.isfinite(matrix[row, column])
                else "NA"
            )
            axis.text(column, row, value_text, ha="center", va="center", fontsize=8)
    axis.set_xticks(range(5), [str(value) for value in FOLDS])
    axis.set_yticks(range(5), [str(value) for value in SEEDS])
    axis.set_xlabel("外层 fold")
    axis.set_ylabel("训练 seed_base")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    return figure


def _fold3_figure(seed_fold: pd.DataFrame) -> Figure:
    data = seed_fold.loc[seed_fold["fold"].eq(3)].sort_values("seed_base")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(data["seed_base"].astype(str), 100 * data["D"], color="#4d9221")
    axis.axhline(5, color="#b2182b", linestyle="--", label="本轮基础损失门槛 5%")
    axis.axhline(
        9.5934, color="#762a83", linestyle=":", label="上一轮外部参考 +9.5934%"
    )
    axis.set_ylabel("验证集状态损失退化（%）")
    axis.set_xlabel("训练 seed_base")
    axis.set_title("外层 fold 3：跨训练 seed 的基础损失退化")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _gain_distribution(seed_level: pd.DataFrame, column: str, title: str) -> Figure:
    values = seed_level.sort_values("seed_base")[column].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 5))
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values):
        axis.boxplot(finite_values, vert=True, widths=0.35, showmeans=True)
    else:
        axis.text(0.5, 0.5, "端点不可验证", transform=axis.transAxes, ha="center")
    jitter = np.linspace(-0.08, 0.08, len(values))
    axis.scatter(1 + jitter[finite], values[finite], color="#2166ac", zorder=3)
    for x, value, seed in zip(
        1 + jitter[finite],
        values[finite],
        seed_level.sort_values("seed_base").loc[finite, "seed_base"],
        strict=True,
    ):
        axis.annotate(
            str(seed), (x, value), xytext=(3, 3), textcoords="offset points", fontsize=8
        )
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.axhline(0.05, color="#b2182b", linestyle="--", linewidth=0.9)
    axis.set_xticks([1], ["五个训练 seed"])
    axis.set_ylabel("G3−G1 效应")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def _pcr_figure(pcr: pd.DataFrame) -> Figure:
    data = pcr.copy()
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DECISION_POINTS))
    width = 0.13
    for index, seed in enumerate(SEEDS):
        part = (
            data.loc[data["seed_base"].eq(seed)]
            .set_index("decision_point")
            .reindex(DECISION_POINTS)
        )
        axis.bar(x + (index - 2) * width, part["delta_auroc"], width, label=str(seed))
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.set_xticks(x, DECISION_POINTS)
    axis.set_ylabel("G3−G1 合并 OOF AUROC")
    axis.set_title("pCR 次要终点：每个训练 seed 的配对 AUROC 差")
    axis.legend(title="训练 seed", ncol=5, frameon=False)
    figure.tight_layout()
    return figure


def _variance_figure(variance: pd.DataFrame) -> Figure:
    endpoint_labels = {
        "dynamic_spearman": "动态 ΔSpearman",
        "static_spearman": "静态 ΔSpearman",
        "base_degradation": "基础损失退化",
    }
    labels = [endpoint_labels.get(value, str(value)) for value in variance["endpoint"]]
    seed_share = variance["seed_share"].to_numpy(dtype=float)
    fold_share = variance["fold_share"].to_numpy(dtype=float)
    interaction = variance["interaction_sampling_share"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x, seed_share, label="训练 seed", color="#2166ac")
    axis.bar(x, fold_share, bottom=seed_share, label="外层 fold", color="#f4a582")
    axis.bar(
        x,
        interaction,
        bottom=seed_share + fold_share,
        label="seed×fold 交互 + 指标采样误差",
        color="#92c5de",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("截断后的方差占比")
    axis.set_title("双因素矩估计方差分解")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _fold_summary_figure(fold_level: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    x = fold_level["fold"].to_numpy(dtype=int)
    axis.errorbar(
        x - 0.05,
        fold_level["static_delta_spearman_mean"],
        yerr=fold_level["static_delta_spearman_sd"],
        fmt="o-",
        capsize=3,
        label="静态 FTV ΔSpearman",
    )
    axis.errorbar(
        x + 0.05,
        fold_level["delta_ftv_delta_spearman_mean"],
        yerr=fold_level["delta_ftv_delta_spearman_sd"],
        fmt="s-",
        capsize=3,
        label="观测 ΔFTV ΔSpearman",
    )
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xlabel("外层 fold")
    axis.set_ylabel("跨训练 seed 均值 ± 样本标准差")
    axis.set_title("fold 层面的静态/动态稳健性")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _safety_figure(stability: pd.DataFrame) -> Figure:
    data = stability.copy()
    data["run"] = (
        data["seed_base"].astype(str)
        + "/F"
        + data["fold"].astype(str)
        + "/"
        + data["model"]
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(data))
    colors = np.where(data["model"].eq("G3"), "#b2182b", "#2166ac")
    axes[0].bar(x, data["selected_epoch"], color=colors)
    axes[0].set_ylabel("选中的 epoch")
    axes[0].set_title("50 次正式运行：选中 epoch 与表征安全性")
    axes[1].bar(x, data["representation_std"], color=colors)
    axes[1].axhline(0.05, color="#000000", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("验证集表征标准差")
    axes[1].set_xticks(x, data["run"], rotation=90, fontsize=7)
    figure.tight_layout()
    return figure


def save_figures(
    conditional: pd.DataFrame,
    seed_fold: pd.DataFrame,
    seed_level: pd.DataFrame,
    pcr: pd.DataFrame,
    variance: pd.DataFrame,
    fold_level: pd.DataFrame,
    stability: pd.DataFrame,
    stage_dir: Path,
    final_dir: Path,
) -> pd.DataFrame:
    _setup_matplotlib()
    stage_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, str, str, Any]] = [
        (
            "01_static_seed_conditional_ci.png",
            "每个训练 seed 的静态 FTV 配对增益",
            "conditional_seed_bootstrap_ci",
            lambda: _seed_ci_figure(
                conditional,
                "dS",
                "静态 FTV：每个训练 seed 的 G3−G1 ΔSpearman 与条件 bootstrap 区间",
            ),
        ),
        (
            "02_dynamic_seed_conditional_ci.png",
            "每个训练 seed 的观测 ΔFTV 配对增益",
            "conditional_seed_bootstrap_ci",
            lambda: _seed_ci_figure(
                conditional,
                "dD",
                "观测 ΔFTV：每个训练 seed 的 G3−G1 ΔSpearman 与条件 bootstrap 区间",
            ),
        ),
        (
            "03_base_degradation_heatmap.png",
            "seed×fold 基础损失退化",
            "seed_fold_effects",
            lambda: _heatmap(seed_fold, "D", "seed×fold 验证集基础损失退化", "D"),
        ),
        (
            "04_dynamic_gain_heatmap.png",
            "seed×fold 动态增益",
            "seed_fold_effects",
            lambda: _heatmap(seed_fold, "dD_sf", "seed×fold 观测 ΔFTV 改善", "dD[s,f]"),
        ),
        (
            "05_fold3_base_degradation.png",
            "fold 3 基础损失退化",
            "seed_fold_effects",
            lambda: _fold3_figure(seed_fold),
        ),
        (
            "06_static_gain_distribution.png",
            "静态增益分布",
            "seed_level_robustness",
            lambda: _gain_distribution(
                seed_level, "dS", "静态 FTV 增益的跨训练 seed 分布"
            ),
        ),
        (
            "07_dynamic_gain_distribution.png",
            "动态增益分布",
            "seed_level_robustness",
            lambda: _gain_distribution(
                seed_level, "dD", "观测 ΔFTV 增益的跨训练 seed 分布"
            ),
        ),
        (
            "08_pcr_secondary_auroc.png",
            "pCR 次要终点配对 AUROC",
            "pcr_secondary_seed_metrics",
            lambda: _pcr_figure(pcr),
        ),
        (
            "09_variance_decomposition.png",
            "seed/fold/交互方差",
            "variance_decomposition",
            lambda: _variance_figure(variance),
        ),
        (
            "10_fold_level_mean_sd.png",
            "fold 层面均值与标准差",
            "fold_level_robustness",
            lambda: _fold_summary_figure(fold_level),
        ),
        (
            "11_selected_epoch_representation_std.png",
            "选中 epoch 与表征标准差",
            "training_stability_seed_fold",
            lambda: _safety_figure(stability),
        ),
    ]
    filenames = [item[0] for item in specs]
    titles = [item[1] for item in specs]
    if len(specs) != 11 or len(set(filenames)) != 11 or len(set(titles)) != 11:
        raise AnalysisInputError("正式 figure 规格必须是 11 个唯一 filename/title")
    rows: list[dict[str, Any]] = []
    for filename, title, source, builder in specs:
        figure = builder()
        path = stage_dir / filename
        figure.savefig(
            path,
            bbox_inches="tight",
            metadata={"Title": title, "Author": "G3 多训练种子正式聚合"},
        )
        plt.close(figure)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.load()
        except Exception as exc:
            raise AnalysisInputError(f"PNG 无法解码: {path}: {exc}") from exc
        if width <= 0 or height <= 0 or path.stat().st_size <= 0:
            raise AnalysisInputError(f"PNG 尺寸/bytes 非法: {path}")
        rows.append(
            {
                "figure": filename,
                "title": title,
                "source": source,
                "path": _portable(final_dir / filename),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
                "width": int(width),
                "height": int(height),
                "decodable": True,
            }
        )
    manifest = pd.DataFrame(rows)
    if (
        len(manifest) != 11
        or manifest["figure"].nunique() != 11
        or manifest["title"].nunique() != 11
        or manifest["path"].nunique() != 11
        or manifest["sha256"].nunique() != 11
    ):
        raise AnalysisInputError("正式 figure manifest filename/title/path/SHA 非唯一")
    return manifest


def _markdown_seed_table(seed_level: pd.DataFrame) -> str:
    lines = [
        "| 训练种子 seed_base | dS | dD | dD_R2 | pCR 纵向 ΔAUROC | 失败 fold 数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_level.sort_values("seed_base").itertuples(index=False):
        lines.append(
            f"| {int(row.seed_base)} | {row.dS:+.4f} | {row.dD:+.4f} | "
            f"{row.dD_R2:+.4f} | {row.pcr_longitudinal_delta_auroc:+.4f} | "
            f"{int(row.failed_fold_count)} |"
        )
    return "\n".join(lines)


def _markdown_pcr_table(pcr_seed_model: pd.DataFrame) -> str:
    summary = (
        pcr_seed_model.groupby(["model", "decision_point"], sort=False)[
            ["auroc", "auprc", "accuracy", "sensitivity", "specificity"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        "| 模型 | 决策点 | AUROC | AUPRC | 准确率 | 敏感度 | 特异度 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.decision_point} | {row.auroc:.4f} | "
            f"{row.auprc:.4f} | {row.accuracy:.4f} | {row.sensitivity:.4f} | "
            f"{row.specificity:.4f} |"
        )
    return "\n".join(lines)


def _markdown_pcr_delta_table(uncertainty: pd.DataFrame) -> str:
    lines = [
        "| 决策点 | ΔAUROC 均值±样本标准差 | 最小/中位/最大 | 95% t-CI | ΔAUPRC 均值±样本标准差 | 最小/中位/最大 | 95% t-CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    endpoints = [
        (
            point,
            f"pcr_{point.replace('-', '_')}_delta_auroc",
            f"pcr_{point.replace('-', '_')}_delta_auprc",
        )
        for point in DECISION_POINTS
    ] + [
        (
            "纵向宏平均（T0-T1/T0-T2）",
            "pcr_longitudinal_delta_auroc",
            "pcr_longitudinal_delta_auprc",
        )
    ]
    by_column = uncertainty.set_index("column")
    for point, auroc_column, auprc_column in endpoints:
        auroc = by_column.loc[auroc_column]
        auprc = by_column.loc[auprc_column]
        lines.append(
            f"| {point} | {auroc['mean']:+.4f}±{auroc['sample_sd']:.4f} | "
            f"{auroc['minimum']:+.4f}/{auroc['median']:+.4f}/{auroc['maximum']:+.4f} | "
            f"[{auroc['t_ci_low']:+.4f}, {auroc['t_ci_high']:+.4f}] | "
            f"{auprc['mean']:+.4f}±{auprc['sample_sd']:.4f} | "
            f"{auprc['minimum']:+.4f}/{auprc['median']:+.4f}/{auprc['maximum']:+.4f} | "
            f"[{auprc['t_ci_low']:+.4f}, {auprc['t_ci_high']:+.4f}] |"
        )
    return "\n".join(lines)


def build_report(
    decision: Mapping[str, Any],
    seed_level: pd.DataFrame,
    uncertainty: pd.DataFrame,
    crossed: pd.DataFrame,
    variance: pd.DataFrame,
    pcr_seed_model: pd.DataFrame,
    pcr_secondary: pd.DataFrame,
) -> str:
    static = uncertainty.loc[uncertainty["column"].eq("dS")].iloc[0]
    dynamic = uncertainty.loc[uncertainty["column"].eq("dD")].iloc[0]
    static_crossed = crossed.loc[crossed["endpoint"].eq("dS")].iloc[0]
    dynamic_crossed = crossed.loc[crossed["endpoint"].eq("dD")].iloc[0]
    primary_variance = variance.loc[variance["endpoint"].eq("dynamic_spearman")].iloc[0]
    gates = decision["gates"]
    conclusion = decision["conclusion"]
    foundation = (
        "值得进入下一阶段" if conclusion == "ROBUST" else "暂不应直接作为下一阶段基础"
    )
    next_step = (
        "可在保持优化协议不变的前提下扩展结构化监督目标。"
        if conclusion == "ROBUST"
        else "应先解决优化与基础损失稳定性，再讨论扩展结构化监督目标。"
    )
    return f"""# G3 多训练种子泛化最终报告

## 结论先行

预注册机械结论为 **{conclusion}**。R1/R2/R3/R4 分别为
`{gates['R1_static_reproducibility']}` / `{gates['R2_dynamic_reproducibility']}` /
`{gates['R3_optimization_safety']}` / `{gates['R4_no_collapse']}`。
pCR 是次要终点，机器字段固定为 `pcr_used_in_decision=false`，没有改变正式结论。

## 五个训练种子的主结果

{_markdown_seed_table(seed_level)}

- 静态主效应均值 dS={float(static['mean']):+.4f}，样本标准差={float(static['sample_sd']):.4f}，
  训练种子层面 95% t-CI [{float(static['t_ci_low']):+.4f}, {float(static['t_ci_high']):+.4f}]。
- 动态主效应均值 dD={float(dynamic['mean']):+.4f}，样本标准差={float(dynamic['sample_sd']):.4f}，
  训练种子层面 95% t-CI [{float(dynamic['t_ci_low']):+.4f}, {float(dynamic['t_ci_high']):+.4f}]。
- 支持性的交叉 bootstrap 区间：dS [{float(static_crossed.ci_low):+.4f}, {float(static_crossed.ci_high):+.4f}]，
  dD [{float(dynamic_crossed.ci_low):+.4f}, {float(dynamic_crossed.ci_high):+.4f}]；它们不进入正式门槛。

## pCR 次要终点

下表是五个训练种子的合并 OOF 指标再取跨种子均值；每个种子、模型和决策点均覆盖 808 名唯一患者。

{_markdown_pcr_table(pcr_seed_model)}

配对的 G3−G1 差值如下；AUROC、AUPRC 及其 2,000 次患者配对 bootstrap 只提供次要证据，不参与 R1–R4 或三级结论。

{_markdown_pcr_delta_table(uncertainty)}

逐训练种子、模型、决策点的 AUROC/AUPRC/准确率/敏感度/特异度见
[pCR 模型指标表](../metrics/final/pcr_seed_model_metrics.csv)，配对差值见
[pCR 次要终点表](../metrics/final/pcr_secondary_seed_metrics.csv) 与
[条件 bootstrap 表](../metrics/final/conditional_seed_bootstrap_ci.csv)。

## 七个冻结问题的回答

1. **静态 FTV 改善是否跨训练种子可重复？** R1={gates['R1_static_reproducibility']}；
   依据为 [训练种子层面端点](../metrics/final/seed_level_robustness.csv)、
   [训练种子 t-CI](../metrics/final/seed_uncertainty.csv) 与
   [逐一剔除种子/fold 重算](../metrics/final/leave_one_out_sensitivity.csv)。
2. **观测 ΔFTV 改善是否跨训练种子可重复？** R2={gates['R2_dynamic_reproducibility']}；
   dD 和描述性 dD_R2 均从每个训练种子五折合并后的 375 名唯一 OOF 患者重算。
3. **上一轮 fold 3 失败是否重复？** 本轮 fold 3 有 {decision['fold3_failures']}/5 个基础损失失败，
   预注册解释为“{decision['fold3_interpretation_cn']}”。上一轮 +9.5934% 只作外部参考。
4. **不稳定性主要来自哪里？** 动态双因素方差分解的主导标签为
   `{primary_variance['dominance']}`；训练种子/fold/交互+采样误差的截断后占比分别为
   {float(primary_variance['seed_share']):.3f}/{float(primary_variance['fold_share']):.3f}/
   {float(primary_variance['interaction_sampling_share']):.3f}。由于每格没有重复，残差不解释为纯交互。
5. **正式类别？** **{conclusion}**，完全由未四舍五入机器表机械得到。
6. **是否值得作为 Factorized Grounded Response State 基础？** {foundation}。
7. **下一步应扩展监督目标，还是先解决优化问题？** {next_step}

## 统计与审计边界

- 正式主不确定性是 5 个训练种子的 t-CI；每个种子的 2,000 次患者 bootstrap 只条件于已拟合模型。
- 5,000 次交叉 bootstrap 同时重采训练种子，并在外层 fold 内同步重采患者，仅作敏感性分析。
- 逐一剔除 fold 的分析会删除相应患者行后重新计算合并 Spearman，未使用 fold rho 的代数平均。
- 公开 `metrics/final` 只有聚合表，不含患者 ID；患者级预测、特征、checkpoint 和训练历史不进入公开结果。
- 完整机器判定见 [decision.json](../metrics/final/decision.json)，输入与图像哈希见各 manifest。

## 注册图

1. [静态端点的训练种子条件区间](../figures/final/01_static_seed_conditional_ci.png)
2. [动态端点的训练种子条件区间](../figures/final/02_dynamic_seed_conditional_ci.png)
3. [基础损失退化热图](../figures/final/03_base_degradation_heatmap.png)
4. [动态增益热图](../figures/final/04_dynamic_gain_heatmap.png)
5. [fold 3 基础损失退化](../figures/final/05_fold3_base_degradation.png)
6. [静态增益分布](../figures/final/06_static_gain_distribution.png)
7. [动态增益分布](../figures/final/07_dynamic_gain_distribution.png)
8. [pCR 次要终点 AUROC](../figures/final/08_pcr_secondary_auroc.png)
9. [方差分解](../figures/final/09_variance_decomposition.png)
10. [fold 层面的稳健性](../figures/final/10_fold_level_mean_sd.png)
11. [选中 epoch 与表征标准差安全性](../figures/final/11_selected_epoch_representation_std.png)
"""


def _coverage_table(
    prediction_manifest: pd.DataFrame,
    history_manifest: pd.DataFrame,
    selection_manifest: pd.DataFrame,
    input_manifest: pd.DataFrame,
    downstream: pd.DataFrame,
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> pd.DataFrame:
    observed = {
        "training_seeds": len(SEEDS),
        "models": len(MODELS),
        "seed_model_fold_cells": len(_expected_grid()),
        "checkpoints": int(input_manifest["kind"].eq("checkpoint").sum()),
        "histories": len(history_manifest),
        "training_selections": len(selection_manifest),
        "features": int(input_manifest["kind"].eq("feature").sum()),
        "probe_prediction_files": int(
            prediction_manifest["kind"].eq("probe_prediction").sum()
        ),
        "pcr_prediction_files": int(
            prediction_manifest["kind"].eq("pcr_prediction").sum()
        ),
        "probe_prediction_rows": len(probe),
        "pcr_prediction_rows": len(pcr),
        "probe_downstream_selections": int(
            downstream["kind"].eq("probe_selection").sum()
        ),
        "pcr_downstream_selections": int(downstream["kind"].eq("pcr_selection").sum()),
        "registered_png": len(figure_manifest),
    }
    expected = {
        "training_seeds": 5,
        "models": 2,
        "seed_model_fold_cells": 50,
        "checkpoints": 50,
        "histories": 50,
        "training_selections": 50,
        "features": 50,
        "probe_prediction_files": 50,
        "pcr_prediction_files": 50,
        "probe_prediction_rows": EXPECTED_PROBE_ROWS,
        "pcr_prediction_rows": EXPECTED_PCR_ROWS,
        "probe_downstream_selections": 50,
        "pcr_downstream_selections": 50,
        "registered_png": 11,
    }
    rows = [
        {
            "asset": key,
            "expected": expected[key],
            "observed": observed[key],
            "passed": observed[key] == expected[key],
        }
        for key in expected
    ]
    output = pd.DataFrame(rows)
    if not output["passed"].all():
        raise AnalysisInputError(
            f"formal coverage 失败: {output.loc[~output['passed']].to_dict('records')}"
        )
    return output


def _public_frame_guard(name: str, frame: pd.DataFrame) -> None:
    forbidden_columns = {
        "patient_id",
        "patientid",
        "y_true",
        "y_pred",
        "probability",
        "predicted_probability",
        "predicted_label",
        "target_change",
        "predicted_change",
    }
    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        for column in frame.columns
    }
    if normalized_columns.intersection(forbidden_columns):
        raise AnalysisInputError(
            f"公开表含患者级敏感列: {name}/{sorted(normalized_columns.intersection(forbidden_columns))}"
        )
    absolute_path = re.compile(
        r"(?:^|[\s\"'=])(?:/(?:home|data|mnt|Users)/|[A-Za-z]:[\\/])"
    )
    secret = re.compile(
        r"(?i)(?:api[_-]?key|secret|password|passwd|bearer\s+[A-Za-z0-9._-]+|"
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    )
    for column in frame.select_dtypes(include="object").columns:
        values = frame[column].dropna().astype(str)
        if values.map(lambda value: bool(absolute_path.search(value))).any():
            raise AnalysisInputError(f"公开表含真实绝对路径: {name}.{column}")
        if values.map(lambda value: bool(secret.search(value))).any():
            raise AnalysisInputError(f"公开表疑似含凭据或 secret: {name}.{column}")


def _remove_target(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _commit_staged(
    staged_targets: Sequence[tuple[Path, Path]], overwrite: bool
) -> None:
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    for _, target in staged_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"拒绝覆盖已有正式输出: {target}")
    try:
        for _, target in staged_targets:
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
                if backup.exists():
                    _remove_target(backup)
                os.replace(target, backup)
                backups.append((target, backup))
        for staged, target in staged_targets:
            os.replace(staged, target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            _remove_target(target)
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _, backup in backups:
            _remove_target(backup)


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    if config.seed != EXPECTED_ANALYSIS_SEED:
        raise AnalysisInputError(f"正式 bootstrap RNG 必须是 {EXPECTED_ANALYSIS_SEED}")
    if config.conditional_replicates != EXPECTED_CONDITIONAL_REPLICATES:
        raise AnalysisInputError("正式 conditional bootstrap 必须 2000 次")
    if config.crossed_replicates != EXPECTED_CROSSED_REPLICATES:
        raise AnalysisInputError("正式 crossed bootstrap 必须 5000 次")
    if config.audit_checkpoints is not True:
        raise AnalysisInputError(
            "正式聚合必须启用全部 50 个 checkpoint tensor/contract 审计"
        )
    locked_paths = {
        "prediction_root": EXPERIMENT_ROOT / "predictions",
        "metric_input_root": EXPERIMENT_ROOT / "metrics",
        "history_root": EXPERIMENT_ROOT / "metrics" / "training" / "formal",
        "checkpoint_root": EXPERIMENT_ROOT / "checkpoints" / "formal",
        "feature_root": EXPERIMENT_ROOT / "features",
        "metric_dir": EXPERIMENT_ROOT / "metrics" / "final",
        "figure_dir": EXPERIMENT_ROOT / "figures" / "final",
        "report_path": EXPERIMENT_ROOT / "reports" / "final_report.md",
    }
    for name, expected in locked_paths.items():
        if Path(getattr(config, name)).resolve() != expected.resolve():
            raise AnalysisInputError(f"冻结计划要求正式 {name}={expected.resolve()}")
    for target in (config.metric_dir, config.figure_dir, config.report_path):
        if target.exists() and not config.overwrite:
            raise FileExistsError(f"拒绝覆盖已有正式输出: {target}")

    freeze_evidence = _validate_freeze_provenance()
    variance_tests = variance_decomposition_self_test()
    probe, pcr, prediction_manifest = discover_predictions(config)
    stability, history_manifest, selection_manifest, input_manifest, downstream = (
        audit_training_assets(config, probe, pcr)
    )
    input_manifest = pd.concat([input_manifest, prediction_manifest], ignore_index=True)
    tables = build_metric_tables(probe, pcr, stability)
    conditional = conditional_bootstrap(
        probe,
        pcr,
        tables["seed_level_robustness"],
        tables["pcr_secondary_seed_metrics"],
        config.conditional_replicates,
        config.seed,
    )
    crossed = crossed_bootstrap(
        probe,
        tables["seed_level_robustness"],
        config.crossed_replicates,
        config.seed,
    )
    leave_one_out = leave_one_out_sensitivity(probe, tables["seed_level_robustness"])
    uncertainty = seed_uncertainty(tables["seed_level_robustness"])
    variance = variance_decomposition_table(tables["seed_fold_effects"])
    decision, gates = compute_decision(
        tables["seed_level_robustness"],
        tables["seed_fold_effects"],
        uncertainty,
        leave_one_out,
        stability,
        variance,
    )

    config.metric_dir.parent.mkdir(parents=True, exist_ok=True)
    config.figure_dir.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_stage = Path(
        tempfile.mkdtemp(
            prefix=f".{config.metric_dir.name}.stage-", dir=config.metric_dir.parent
        )
    )
    figure_stage = Path(
        tempfile.mkdtemp(
            prefix=f".{config.figure_dir.name}.stage-", dir=config.figure_dir.parent
        )
    )
    report_stage_dir = Path(
        tempfile.mkdtemp(prefix=".report.stage-", dir=config.report_path.parent)
    )
    report_stage = report_stage_dir / config.report_path.name
    try:
        figure_manifest = save_figures(
            conditional,
            tables["seed_fold_effects"],
            tables["seed_level_robustness"],
            tables["pcr_secondary_seed_metrics"],
            variance,
            tables["fold_level_robustness"],
            stability,
            figure_stage,
            config.figure_dir,
        )
        coverage = _coverage_table(
            prediction_manifest,
            history_manifest,
            selection_manifest,
            input_manifest,
            downstream,
            probe,
            pcr,
            figure_manifest,
        )
        issues = pd.DataFrame(columns=("severity", "asset", "issue"))
        public_tables: dict[str, pd.DataFrame] = {
            "training_stability_seed_fold.csv": stability,
            "probe_seed_cell_metrics.csv": tables["probe_seed_cell_metrics"],
            "probe_seed_fold_cell_metrics.csv": tables["probe_seed_fold_cell_metrics"],
            "seed_fold_effects.csv": tables["seed_fold_effects"],
            "seed_level_robustness.csv": tables["seed_level_robustness"],
            "fold_level_robustness.csv": tables["fold_level_robustness"],
            "seed_uncertainty.csv": uncertainty,
            "conditional_seed_bootstrap_ci.csv": conditional,
            "crossed_bootstrap_ci.csv": crossed,
            "leave_one_out_sensitivity.csv": leave_one_out,
            "variance_decomposition.csv": variance,
            "pcr_secondary_seed_metrics.csv": tables["pcr_secondary_seed_metrics"],
            "pcr_seed_model_metrics.csv": tables["pcr_seed_model_metrics"],
            "decision_gates.csv": gates,
            "coverage.csv": coverage,
            "issues.csv": issues,
            "input_manifest.csv": input_manifest,
            "history_manifest.csv": history_manifest,
            "selection_manifest.csv": selection_manifest,
            "prediction_manifest.csv": prediction_manifest,
            "figure_manifest.csv": figure_manifest,
            "downstream_selection_audit.csv": downstream,
        }
        for name, frame in public_tables.items():
            _public_frame_guard(name, frame)
            _write_csv(metric_stage / name, frame)
        decision.update(
            {
                "formal_analysis": True,
                "bootstrap_rng_seed": config.seed,
                "conditional_bootstrap_replicates": config.conditional_replicates,
                "crossed_bootstrap_replicates": config.crossed_replicates,
                "analysis_source_sha256": file_sha256(Path(__file__)),
            }
        )
        _write_json(metric_stage / "decision.json", decision)
        acceptance = {
            "schema_version": SCHEMA_VERSION,
            "formal_analysis": True,
            "public_tables_contain_patient_rows": False,
            "prediction_rows_recomputed_in_memory": {
                "probe": len(probe),
                "pcr": len(pcr),
            },
            "variance_synthetic_tests": variance_tests,
            "decision_recomputed_from_unrounded_tables": True,
            "pcr_used_in_decision": False,
            "registered_issues": len(issues),
            "coverage_all_passed": bool(coverage["passed"].all()),
            "figure_count": len(figure_manifest),
            "analysis_source_sha256": file_sha256(Path(__file__)),
            "plan_sha256": freeze_evidence["plan_sha256"],
            "plan_freeze_sha256": freeze_evidence["plan_freeze_sha256"],
            "source_freeze_sha256": freeze_evidence["source_freeze_sha256"],
            "source_implementation_sha256": freeze_evidence[
                "source_implementation_sha256"
            ],
            "preflight_sha256": freeze_evidence["preflight_sha256"],
            "smoke_gate_sha256": freeze_evidence["smoke_gate_sha256"],
            "training_implementation_sha256": freeze_evidence[
                "training_implementation_sha256"
            ],
            "formal_namespace_locked": config.metric_dir
            == (EXPERIMENT_ROOT / "metrics" / "final").resolve()
            and config.figure_dir == (EXPERIMENT_ROOT / "figures" / "final").resolve()
            and config.report_path
            == (EXPERIMENT_ROOT / "reports" / "final_report.md").resolve(),
            "prediction_schema_strict": True,
            "downstream_protocol_independently_audited": True,
            "canonical_feature_rows_independently_verified": True,
            "probe_targets_scalers_readouts_predictions_independently_recomputed": True,
            "pcr_scalers_readouts_thresholds_predictions_independently_recomputed": True,
            "paired_bootstrap_draws_synchronized": True,
        }
        _write_json(metric_stage / "analysis_acceptance_evidence.json", acceptance)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "formal_analysis": True,
            "conclusion": decision["conclusion"],
            "seeds": list(SEEDS),
            "models": list(MODELS),
            "folds": list(FOLDS),
            "probe_rows_consumed_in_memory": len(probe),
            "pcr_rows_consumed_in_memory": len(pcr),
            "public_patient_rows": 0,
            "conditional_bootstrap_replicates": config.conditional_replicates,
            "crossed_bootstrap_replicates": config.crossed_replicates,
            "bootstrap_rng_seed": config.seed,
            "seed_t_ci_is_formal_gate": True,
            "pcr_used_in_decision": False,
            "figures": len(figure_manifest),
            "registered_issues": len(issues),
            "analysis_source_sha256": file_sha256(Path(__file__)),
            "plan_sha256": freeze_evidence["plan_sha256"],
            "plan_freeze_sha256": freeze_evidence["plan_freeze_sha256"],
            "source_freeze_sha256": freeze_evidence["source_freeze_sha256"],
            "source_implementation_sha256": freeze_evidence[
                "source_implementation_sha256"
            ],
            "preflight_sha256": freeze_evidence["preflight_sha256"],
            "smoke_gate_sha256": freeze_evidence["smoke_gate_sha256"],
            "training_implementation_sha256": freeze_evidence[
                "training_implementation_sha256"
            ],
            "downstream_predictions_independently_recomputed": True,
        }
        _write_json(metric_stage / "aggregation_summary.json", summary)
        report_stage.write_text(
            build_report(
                decision,
                tables["seed_level_robustness"],
                uncertainty,
                crossed,
                variance,
                tables["pcr_seed_model_metrics"],
                tables["pcr_secondary_seed_metrics"],
            ),
            encoding="utf-8",
        )
        _commit_staged(
            (
                (metric_stage, config.metric_dir),
                (figure_stage, config.figure_dir),
                (report_stage, config.report_path),
            ),
            config.overwrite,
        )
        report_stage_dir.rmdir()
    except Exception:
        for path in (metric_stage, figure_stage, report_stage_dir):
            if path.exists():
                _remove_target(path)
        raise
    return summary


def _synthetic_inputs(
    seed: int = 91,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe_rows: list[dict[str, Any]] = []
    pcr_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for seed_index, seed_base in enumerate(SEEDS):
        for fold in FOLDS:
            rng = np.random.default_rng(seed + seed_index * 100 + fold)
            for patient_index in range(12):
                patient_id = f"m{fold:01d}_{patient_index:03d}"
                latent = patient_index / 11 + fold * 0.07
                for task, cells in (("static", TIMEPOINTS), ("change", TRANSITIONS)):
                    for cell_index, cell in enumerate(cells):
                        target = (
                            latent
                            + 0.15 * cell_index
                            + (0.1 if task == "change" else 0.0)
                        )
                        baseline = 0.5 + 0.15 * cell_index
                        g1 = rng.normal(0.45, 0.45) + 0.12 * target
                        g3 = target + rng.normal(0, 0.045)
                        for model, prediction in (("G1", g1), ("G3", g3)):
                            probe_rows.append(
                                {
                                    "patient_id": patient_id,
                                    "fold": fold,
                                    "seed_base": seed_base,
                                    "model": model,
                                    "task": task,
                                    "cell": cell,
                                    "y_true": target,
                                    "y_pred": prediction,
                                    "b0_prediction": baseline,
                                }
                            )
            for patient_index in range(20):
                patient_id = f"p{fold:01d}_{patient_index:03d}"
                target = int(patient_index % 2)
                for point_index, point in enumerate(DECISION_POINTS):
                    g1_probability = float(
                        np.clip(
                            0.43 + 0.14 * target + rng.normal(0, 0.12), 0.001, 0.999
                        )
                    )
                    g3_probability = float(
                        np.clip(
                            0.12 + 0.76 * target + rng.normal(0, 0.04), 0.001, 0.999
                        )
                    )
                    for model, probability in (
                        ("G1", g1_probability),
                        ("G3", g3_probability),
                    ):
                        pcr_rows.append(
                            {
                                "patient_id": patient_id,
                                "fold": fold,
                                "seed_base": seed_base,
                                "model": model,
                                "decision_point": point,
                                "y_true": target,
                                "probability": probability,
                                "predicted_label": int(probability >= 0.5),
                            }
                        )
            g1_loss = 0.10 + 0.004 * fold + 0.001 * seed_index
            for model in MODELS:
                g3 = model == "G3"
                loss = g1_loss * (1.01 if g3 else 1.0)
                stability_rows.append(
                    {
                        "seed_base": seed_base,
                        "fold": fold,
                        "effective_seed": seed_base + fold,
                        "model": model,
                        "selected_epoch": 3,
                        "selection_mode": "primary",
                        "val_state_loss": loss,
                        "val_ftv_loss": 0.2 if g3 else 0.0,
                        "representation_std": 0.4,
                        "lambda_ftv": 0.25 if g3 else 0.0,
                        "selected_scalars_finite": True,
                        "checkpoint_tensors_finite": True,
                        "checkpoint_tensor_count": 1,
                        "feature_finite": True,
                        "no_collapse": True,
                        "architecture_contract_verified": True,
                        "test_data_used": False,
                        "checkpoint_sha256": "0" * 64,
                        "feature_sha256": "1" * 64,
                        "shared_initialization_sha256": f"{seed_base + fold:064x}",
                        "paired_baseline": "G1" if g3 else pd.NA,
                        "base_degradation_fraction": 0.01 if g3 else math.nan,
                        "base_pass": True if g3 else pd.NA,
                    }
                )
    stability = pd.DataFrame(stability_rows)
    stability["base_pass"] = stability["base_pass"].astype("boolean")
    return pd.DataFrame(probe_rows), pd.DataFrame(pcr_rows), stability


def prediction_contract_negative_self_test() -> dict[str, bool]:
    """不写文件地证明 pCR schema/label/protocol 篡改会被聚合输入层拒绝。"""

    rows: list[dict[str, Any]] = []
    for point in DECISION_POINTS:
        schema = PCR_FEATURE_SCHEMAS[point]
        row: dict[str, Any] = {
            "patient_id": "synthetic-patient",
            "seed_base": 2026,
            "fold": 0,
            "effective_seed": 2026,
            "split": "test",
            "model": "G1",
            "decision_point": point,
            "y_true": 0,
            "probability": 0.7,
            "predicted_label": 1,
            "threshold": 0.5,
            "penalty": "l1",
            "C": 0.001,
            "readout": "class-balanced LogisticRegression",
            "class_weight": "balanced",
            "feature_schema": schema,
            "feature_schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
            "feature_dim": PCR_FEATURE_DIMS[point],
            "val_auroc": 0.7,
            "val_auprc": 0.6,
            "val_youden": 0.2,
            "source_feature_file": "/synthetic/feature.npz",
            "source_feature_sha256": "1" * 64,
            "feature_extractor_sha256": "2" * 64,
            "source_checkpoint": "/synthetic/best.pt",
            "source_checkpoint_sha256": "3" * 64,
            "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
            "canonical_patient_order_sha256": "4" * 64,
            "canonical_patient_label_sha256": "5" * 64,
            "test_feature_matrix_constructed_after_selection_lock": True,
            "test_prediction_guard_enforced": True,
            "test_predict_proba_call_count": 1,
        }
        row.update({column: False for column in PCR_FALSE_FLAGS})
        rows.append(row)
    valid = pd.DataFrame(rows, columns=PCR_PREDICTION_COLUMNS)

    def accepted(frame: pd.DataFrame) -> bool:
        original = pd.read_csv
        pd.read_csv = lambda *args, **kwargs: frame.copy()  # type: ignore[method-assign]
        try:
            _normalise_pcr_file(Path("/never-read.csv"), 2026, "G1", 0)
        except (AnalysisInputError, KeyError, TypeError, ValueError):
            return False
        finally:
            pd.read_csv = original  # type: ignore[method-assign]
        return True

    cases: dict[str, pd.DataFrame] = {}
    fractional_truth = valid.copy()
    fractional_truth["y_true"] = 0.5
    cases["fractional_y_true_rejected"] = fractional_truth
    nonbinary_label = valid.copy()
    nonbinary_label["predicted_label"] = 7
    cases["nonbinary_predicted_label_rejected"] = nonbinary_label
    invalid_threshold = valid.copy()
    invalid_threshold["threshold"] = 9.0
    cases["out_of_range_threshold_rejected"] = invalid_threshold
    inconsistent_label = valid.copy()
    inconsistent_label["predicted_label"] = 0
    cases["threshold_label_mismatch_rejected"] = inconsistent_label
    illegal_c = valid.copy()
    illegal_c["C"] = 2.0
    cases["illegal_C_rejected"] = illegal_c
    bad_schema_hash = valid.copy()
    bad_schema_hash["feature_schema_sha256"] = "f" * 64
    cases["schema_hash_mismatch_rejected"] = bad_schema_hash
    missing_penalty = valid.drop(columns="penalty")
    cases["missing_penalty_column_rejected"] = missing_penalty
    checks = {"valid_pcr_contract_accepted": accepted(valid)}
    checks.update({name: not accepted(frame) for name, frame in cases.items()})
    try:
        _strict_bool(pd.Series(["FAIL"]), "negative.bool")
        checks["invalid_boolean_rejected"] = False
    except AnalysisInputError:
        checks["invalid_boolean_rejected"] = True
    return checks


def run_self_test() -> dict[str, Any]:
    probe, pcr, stability = _synthetic_inputs()
    tables = build_metric_tables(probe, pcr, stability)
    conditional = conditional_bootstrap(
        probe,
        pcr,
        tables["seed_level_robustness"],
        tables["pcr_secondary_seed_metrics"],
        40,
        EXPECTED_ANALYSIS_SEED,
    )
    crossed = crossed_bootstrap(
        probe,
        tables["seed_level_robustness"],
        60,
        EXPECTED_ANALYSIS_SEED,
    )
    leave_one_out = leave_one_out_sensitivity(probe, tables["seed_level_robustness"])
    uncertainty = seed_uncertainty(tables["seed_level_robustness"])
    variance = variance_decomposition_table(tables["seed_fold_effects"])
    variance_checks = variance_decomposition_self_test()
    decision, gates = compute_decision(
        tables["seed_level_robustness"],
        tables["seed_fold_effects"],
        uncertainty,
        leave_one_out,
        stability,
        variance,
    )
    with tempfile.TemporaryDirectory(
        prefix="g3-multiseed-analysis-selftest-"
    ) as temporary:
        temporary_root = Path(temporary)
        figure_manifest = save_figures(
            conditional,
            tables["seed_fold_effects"],
            tables["seed_level_robustness"],
            tables["pcr_secondary_seed_metrics"],
            variance,
            tables["fold_level_robustness"],
            stability,
            temporary_root / "figures",
            temporary_root / "final_figures",
        )
        report = build_report(
            decision,
            tables["seed_level_robustness"],
            uncertainty,
            crossed,
            variance,
            tables["pcr_seed_model_metrics"],
            tables["pcr_secondary_seed_metrics"],
        )
    broken_probe = probe.copy()
    broken_mask = (
        broken_probe["seed_base"].eq(SEEDS[0])
        & broken_probe["model"].eq("G3")
        & broken_probe["task"].eq("static")
        & broken_probe["cell"].eq("T0")
    )
    broken_probe.loc[broken_mask, "y_pred"] = 0.0
    broken_tables = build_metric_tables(broken_probe, pcr, stability)
    broken_loo = leave_one_out_sensitivity(
        broken_probe, broken_tables["seed_level_robustness"]
    )
    broken_uncertainty = seed_uncertainty(broken_tables["seed_level_robustness"])
    broken_variance = variance_decomposition_table(broken_tables["seed_fold_effects"])
    broken_decision, _ = compute_decision(
        broken_tables["seed_level_robustness"],
        broken_tables["seed_fold_effects"],
        broken_uncertainty,
        broken_loo,
        stability,
        broken_variance,
    )
    prediction_contract_checks = prediction_contract_negative_self_test()
    checks = {
        "seed_endpoints_five": len(tables["seed_level_robustness"]) == 5,
        "seed_fold_grid_25": len(tables["seed_fold_effects"]) == 25,
        "conditional_contains_primary_and_pcr": set(("dS", "dD", "dD_R2")).issubset(
            set(conditional["endpoint"])
        )
        and conditional["cohort"].eq("pCR").any(),
        "crossed_three_endpoints": set(crossed["endpoint"]) == {"dS", "dD", "dD_R2"},
        "lofo_recomputed": leave_one_out.loc[
            leave_one_out["scope"].str.startswith("leave_one_fold")
        ]["spearman_recomputed_from_patient_rows"].all(),
        "gate_rows_present": {"R1", "R2", "R3", "R4"}.issubset(set(gates["gate"])),
        "pcr_excluded_from_decision": decision["pcr_used_in_decision"] is False,
        "eleven_png_decode": len(figure_manifest) == 11
        and figure_manifest["decodable"].astype(bool).all(),
        "chinese_report_generated": "七个冻结问题" in report
        and decision["conclusion"] in report
        and "pCR 次要终点" in report
        and "AUPRC" in report
        and "准确率" in report
        and "敏感度" in report
        and "特异度" in report,
        "nonfinite_endpoint_mechanically_not_robust": broken_decision["conclusion"]
        == "NOT ROBUST"
        and not bool(
            broken_uncertainty.loc[
                broken_uncertainty["column"].eq("dS"), "verifiable"
            ].iloc[0]
        ),
        **prediction_contract_checks,
        **variance_checks,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(f"analysis synthetic self-test failed: {checks}")
    return {
        "status": "ok",
        "checks": checks,
        "synthetic_conclusion": decision["conclusion"],
        "conditional_replicates_used": 40,
        "crossed_replicates_used": 60,
        "formal_aggregation_run": False,
    }
