"""固定 checkpoint/batch 上独立提取 JEPA base 与 raw FTV gradients。"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .assets import (
    AuditDataContext,
    checkpoint_path,
    checkpoint_state_fingerprint,
    fold_transform,
    load_data_context,
    patient_records,
)
from .batches import PUBLIC_MANIFEST, validate_manifests
from .contracts import (
    AUDIT_ROOT,
    AUDIT_SEED,
    BATCHES_PER_SPLIT,
    BATCH_SIZE,
    GROUPS,
    LAMBDA_FTV,
    SEED_BASES,
    SOURCE_SRC,
    SPLITS,
    atomic_csv,
    derive_stochastic_seed,
    file_sha256,
    repo_relative,
)
from .freeze import SOURCE_CONTRACT
from .source_contract import assert_source_contract

if str(SOURCE_SRC) not in sys.path:
    sys.path.insert(0, str(SOURCE_SRC))

from dgrs.data import LongitudinalDGRSDataset  # noqa: E402
from dgrs.model import DGRSWorldModel, load_checkpoint_for_evaluation  # noqa: E402
from dgrs.training import DGRSObjective  # noqa: E402


EXPECTED_GROUP_COUNTS = {
    "encoder_stage_1": (7, 10_112),
    "encoder_stage_2": (7, 42_112),
    "encoder_stage_3": (7, 168_192),
    "encoder_stage_4": (7, 672_256),
    "encoder_overall": (28, 892_672),
    "response_projection": (4, 25_152),
    "all_shared": (32, 917_824),
}

BATCH_GRADIENT_COLUMNS = (
    "schema_version",
    "seed_base",
    "fold",
    "effective_seed",
    "base_gate",
    "base_degradation",
    "checkpoint_kind",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "split",
    "batch_id",
    "batch_index",
    "ordered_members_hmac_sha256",
    "n_total",
    "n_ftv_available",
    "n_ftv_valid_visits",
    "group",
    "parameter_tensors",
    "parameter_count",
    "base_gradient_none_count",
    "ftv_gradient_none_count",
    "base_gradient_norm",
    "ftv_gradient_norm_raw",
    "weighted_ftv_gradient_norm",
    "gradient_dot_raw",
    "gradient_cosine",
    "weighted_gradient_norm_ratio",
    "base_descent_margin",
    "ftv_descent_margin",
    "negative_cosine",
    "strong_negative_cosine",
    "very_strong_negative_cosine",
    "base_descent_failure",
    "base_objective",
    "state_loss",
    "sigreg_loss",
    "ftv_loss_raw",
    "lambda_ftv",
    "stochastic_seed",
    "model_mode",
    "deterministic_algorithms",
    "paired_forward_outputs_exact",
    "component_forward_backward_count",
    "optimizer_created",
    "optimizer_step",
    "pcr_signal_used",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "source_contract_sha256",
    "public_manifest_sha256",
    "contains_patient_ids",
)


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def _set_stochastic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _outputs_match(left: Any, right: Any) -> bool:
    tensor_fields = (
        "response_state",
        "online_state",
        "target_response_state",
        "target_state",
        "target_next",
        "predicted_next",
        "ftv_prediction",
    )
    for name in tensor_fields:
        first = getattr(left, name)
        second = getattr(right, name)
        if first is None or second is None:
            if first is not None or second is not None:
                return False
        elif not torch.equal(first.detach(), second.detach()):
            return False
    return True


def _shared_named_parameters(model: DGRSWorldModel) -> dict[str, torch.nn.Parameter]:
    result = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("encoder.") or name.startswith("response_projection."))
    }
    if (
        len(result) != 32
        or sum(parameter.numel() for parameter in result.values()) != 917_824
    ):
        raise ValueError("shared gradient parameter contract 漂移")
    return result


def parameter_groups(model: DGRSWorldModel) -> dict[str, tuple[str, ...]]:
    shared = _shared_named_parameters(model)
    groups = {
        "encoder_stage_1": tuple(
            name for name in shared if name.startswith("encoder.features.0.")
        ),
        "encoder_stage_2": tuple(
            name for name in shared if name.startswith("encoder.features.1.")
        ),
        "encoder_stage_3": tuple(
            name for name in shared if name.startswith("encoder.features.2.")
        ),
        "encoder_stage_4": tuple(
            name for name in shared if name.startswith("encoder.features.3.")
        ),
        "encoder_overall": tuple(
            name for name in shared if name.startswith("encoder.")
        ),
        "response_projection": tuple(
            name for name in shared if name.startswith("response_projection.")
        ),
        "all_shared": tuple(shared),
    }
    if set(groups) != set(GROUPS):
        raise ValueError("gradient groups 与冻结合同不一致")
    for name, parameter_names in groups.items():
        observed = (
            len(parameter_names),
            sum(shared[item].numel() for item in parameter_names),
        )
        if observed != EXPECTED_GROUP_COUNTS[name]:
            raise ValueError(f"gradient group {name} count 漂移: {observed}")
    return groups


def _excluded_gradient_assertions(model: DGRSWorldModel, component: str) -> None:
    named = dict(model.named_parameters())
    if component == "base":
        unexpected = [
            name
            for name, parameter in named.items()
            if name.startswith("ftv_head.") and parameter.grad is not None
        ]
    elif component == "ftv":
        unexpected = [
            name
            for name, parameter in named.items()
            if (name.startswith("projector.") or name.startswith("transition."))
            and parameter.grad is not None
        ]
    else:
        raise ValueError("gradient component 非法")
    target_grad = [
        name
        for name, parameter in named.items()
        if name.startswith("target_") and parameter.grad is not None
    ]
    if unexpected or target_grad:
        raise ValueError(
            f"{component} gradient graph 越界: unexpected={unexpected[:3]}, target={target_grad[:3]}"
        )


def _component_sums(
    names: Iterable[str],
    base_grads: Mapping[str, torch.Tensor | None],
    ftv_grads: Mapping[str, torch.Tensor | None],
) -> dict[str, Any]:
    names = tuple(names)
    base_missing = [name for name in names if base_grads[name] is None]
    ftv_missing = [name for name in names if ftv_grads[name] is None]
    if base_missing or ftv_missing:
        return {
            "base_missing": len(base_missing),
            "ftv_missing": len(ftv_missing),
            "base_sq": math.nan,
            "ftv_sq": math.nan,
            "dot": math.nan,
        }
    base_sq = 0.0
    ftv_sq = 0.0
    dot = 0.0
    for name in names:
        left = base_grads[name]
        right = ftv_grads[name]
        assert left is not None and right is not None
        left64 = left.detach().reshape(-1).to(dtype=torch.float64)
        right64 = right.detach().reshape(-1).to(dtype=torch.float64)
        base_sq += float(torch.dot(left64, left64))
        ftv_sq += float(torch.dot(right64, right64))
        dot += float(torch.dot(left64, right64))
    return {
        "base_missing": 0,
        "ftv_missing": 0,
        "base_sq": base_sq,
        "ftv_sq": ftv_sq,
        "dot": dot,
    }


def gradient_geometry(base_sq: float, ftv_sq: float, dot: float) -> dict[str, float]:
    if (
        not all(math.isfinite(value) for value in (base_sq, ftv_sq, dot))
        or base_sq <= 0
        or ftv_sq <= 0
    ):
        return {
            "base_gradient_norm": math.nan,
            "ftv_gradient_norm_raw": math.nan,
            "weighted_ftv_gradient_norm": math.nan,
            "gradient_dot_raw": dot,
            "gradient_cosine": math.nan,
            "weighted_gradient_norm_ratio": math.nan,
            "base_descent_margin": math.nan,
            "ftv_descent_margin": math.nan,
        }
    base_norm = math.sqrt(base_sq)
    ftv_norm = math.sqrt(ftv_sq)
    cosine = dot / (base_norm * ftv_norm)
    ratio = LAMBDA_FTV * ftv_norm / base_norm
    m_base = 1.0 + LAMBDA_FTV * dot / base_sq
    m_ftv = (dot + LAMBDA_FTV * ftv_sq) / (LAMBDA_FTV * ftv_sq)
    return {
        "base_gradient_norm": base_norm,
        "ftv_gradient_norm_raw": ftv_norm,
        "weighted_ftv_gradient_norm": LAMBDA_FTV * ftv_norm,
        "gradient_dot_raw": dot,
        "gradient_cosine": cosine,
        "weighted_gradient_norm_ratio": ratio,
        "base_descent_margin": m_base,
        "ftv_descent_margin": m_ftv,
    }


def _load_split_batches(
    fold: int,
    split: str,
    device: torch.device,
    context: AuditDataContext,
    public: pd.DataFrame,
    private: pd.DataFrame,
) -> tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    selected_private = private.loc[
        private["fold"].eq(fold) & private["split"].eq(split)
    ].sort_values(["batch_index", "position"])
    selected_public = public.loc[
        public["fold"].eq(fold) & public["split"].eq(split)
    ].sort_values("batch_index")
    if (
        len(selected_private) != BATCHES_PER_SPLIT * BATCH_SIZE
        or len(selected_public) != BATCHES_PER_SPLIT
    ):
        raise ValueError(f"audit batch coverage 错误: {fold}/{split}")
    ids = selected_private["patient_id"].astype(str).tolist()
    records = patient_records(context, ids)
    transform = fold_transform(context, fold)
    transformed = transform.transform_all(context.raw_ftv)
    dataset = LongitudinalDGRSDataset(records, transformed, context.raw_ftv)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        drop_last=False,
    )
    return loader, selected_public, selected_private


def _gate_for(seed_base: int, fold: int) -> tuple[str, float]:
    path = AUDIT_ROOT / "metrics" / "run_level_existing_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError("必须先生成 run_level_existing_metrics.csv")
    frame = pd.read_csv(path)
    row = frame.loc[frame["seed_base"].eq(seed_base) & frame["fold"].eq(fold)]
    if len(row) != 1:
        raise ValueError("run-level base gate key 缺失")
    return str(row.iloc[0]["base_gate"]), float(row.iloc[0]["base_degradation"])


def output_path(seed_base: int, fold: int, checkpoint_kind: str) -> Path:
    return (
        AUDIT_ROOT
        / "metrics"
        / "raw"
        / f"gradient_seed_{seed_base}_fold_{fold}_{checkpoint_kind}.csv"
    )


def extract_run(
    seed_base: int,
    fold: int,
    checkpoint_kind: str,
    device_name: str,
    *,
    overwrite: bool = False,
) -> Path:
    if seed_base not in SEED_BASES or checkpoint_kind not in {"selected", "last"}:
        raise ValueError("gradient extraction key 非法")
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    _configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    kind = "best" if checkpoint_kind == "selected" else "last"
    checkpoint = checkpoint_path(seed_base, fold, "g3", kind)
    asset_manifest = pd.read_csv(AUDIT_ROOT / "metrics" / "asset_manifest.csv")
    registered = asset_manifest.loc[
        asset_manifest["seed_base"].eq(seed_base)
        & asset_manifest["fold"].eq(fold)
        & asset_manifest["checkpoint_kind"].eq(checkpoint_kind)
    ]
    if (
        len(registered) != 1
        or str(registered.iloc[0]["checkpoint_sha256"]) != file_sha256(checkpoint)
        or str(registered.iloc[0]["checkpoint"]) != repo_relative(checkpoint)
    ):
        raise ValueError("requested checkpoint 不在冻结 asset manifest")
    model, payload = load_checkpoint_for_evaluation(checkpoint, device)
    if (
        payload.get("model_name") != "G3"
        or float(payload["loss_config"]["lambda_ftv"]) != LAMBDA_FTV
    ):
        raise ValueError("checkpoint model/lambda contract 漂移")
    model.train()
    objective = DGRSObjective(
        model_name="G3",
        lambda_ftv=LAMBDA_FTV,
        sigreg_weight=float(payload["loss_config"]["sigreg"]),
        sigreg_projections=int(payload["loss_config"]["sigreg_projections"]),
        step_weights=tuple(
            float(value) for value in payload["loss_config"]["step_weights"]
        ),
    ).to(device)
    objective.train()
    shared = _shared_named_parameters(model)
    groups = parameter_groups(model)
    before = checkpoint_state_fingerprint({"state_dict": model.state_dict()})
    gate, degradation = _gate_for(seed_base, fold)
    rows: list[dict[str, Any]] = []
    checkpoint_sha = file_sha256(checkpoint)
    public_manifest_sha = file_sha256(PUBLIC_MANIFEST)
    context = load_data_context()
    public_manifest, private_manifest = validate_manifests(
        context, verify_checkpoint_pools=False
    )
    for split in SPLITS:
        loader, public, private = _load_split_batches(
            fold, split, device, context, public_manifest, private_manifest
        )
        public_by_index = {
            int(row.batch_index): row for row in public.itertuples(index=False)
        }
        private_by_index = {
            int(batch_index): batch.sort_values("position")["patient_id"]
            .astype(str)
            .tolist()
            for batch_index, batch in private.groupby("batch_index", sort=True)
        }
        for batch_index, batch in enumerate(loader):
            if batch_index not in range(BATCHES_PER_SPLIT):
                raise ValueError("DataLoader 产生额外 audit batch")
            manifest = public_by_index[batch_index]
            if list(map(str, batch["patient_id"])) != private_by_index[batch_index]:
                raise ValueError(
                    "DataLoader patient order 与 private fixed batch 不一致"
                )
            image = batch["image"].to(device, non_blocking=True)
            ftv_target = batch["ftv_target"].to(device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device, non_blocking=True)
            if tuple(image.shape) != (BATCH_SIZE, 4, 7, 32, 96, 96):
                raise ValueError(f"audit batch image shape 错误: {tuple(image.shape)}")
            if not bool(torch.as_tensor(batch["pcr"]).eq(-1).all()) or not bool(
                torch.as_tensor(batch["label_pcr"]).eq(-1).all()
            ):
                raise ValueError("audit batch 意外加载 pCR label")
            grounded = int(ftv_mask.any(dim=1).sum())
            if grounded != int(manifest.n_ftv_available) or grounded < 8:
                raise ValueError("audit batch grounded count 不一致")
            stochastic_seed = derive_stochastic_seed(fold, split, batch_index)

            model.zero_grad(set_to_none=True)
            _set_stochastic_seed(stochastic_seed)
            base_output = model(image, None)
            _, base_stats = objective(base_output, ftv_target, ftv_mask)
            base_component = base_stats["_base_component"]
            if not bool(torch.isfinite(base_component)):
                raise FloatingPointError("base component nonfinite")
            base_component.backward()
            _excluded_gradient_assertions(model, "base")
            base_grads = {
                name: (
                    None if parameter.grad is None else parameter.grad.detach().clone()
                )
                for name, parameter in shared.items()
            }

            model.zero_grad(set_to_none=True)
            _set_stochastic_seed(stochastic_seed)
            ftv_output = model(image, None)
            if not _outputs_match(base_output, ftv_output):
                raise ValueError("相同 stochastic seed 的双 forward 输出不一致")
            _, ftv_stats = objective(ftv_output, ftv_target, ftv_mask)
            ftv_component = ftv_stats["_ftv_component_raw"]
            if not bool(torch.isfinite(ftv_component)):
                raise FloatingPointError("FTV component nonfinite")
            ftv_component.backward()
            _excluded_gradient_assertions(model, "ftv")
            ftv_grads = {
                name: (
                    None if parameter.grad is None else parameter.grad.detach().clone()
                )
                for name, parameter in shared.items()
            }

            for group_name in GROUPS:
                names = groups[group_name]
                sums = _component_sums(names, base_grads, ftv_grads)
                geometry = gradient_geometry(
                    sums["base_sq"], sums["ftv_sq"], sums["dot"]
                )
                cosine = geometry["gradient_cosine"]
                margin = geometry["base_descent_margin"]
                rows.append(
                    {
                        "schema_version": 1,
                        "seed_base": seed_base,
                        "fold": fold,
                        "effective_seed": seed_base + fold,
                        "base_gate": gate,
                        "base_degradation": degradation,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint_epoch": int(payload["epoch"]),
                        "checkpoint_sha256": checkpoint_sha,
                        "split": split,
                        "batch_id": str(manifest.batch_id),
                        "batch_index": batch_index,
                        "ordered_members_hmac_sha256": str(
                            manifest.ordered_members_hmac_sha256
                        ),
                        "n_total": BATCH_SIZE,
                        "n_ftv_available": grounded,
                        "n_ftv_valid_visits": int(ftv_mask.sum()),
                        "group": group_name,
                        "parameter_tensors": len(names),
                        "parameter_count": sum(shared[name].numel() for name in names),
                        "base_gradient_none_count": int(sums["base_missing"]),
                        "ftv_gradient_none_count": int(sums["ftv_missing"]),
                        **geometry,
                        "negative_cosine": bool(math.isfinite(cosine) and cosine < 0),
                        "strong_negative_cosine": bool(
                            math.isfinite(cosine) and cosine < -0.1
                        ),
                        "very_strong_negative_cosine": bool(
                            math.isfinite(cosine) and cosine < -0.25
                        ),
                        "base_descent_failure": bool(
                            math.isfinite(margin) and margin < 0
                        ),
                        "base_objective": float(base_component.detach()),
                        "state_loss": float(base_stats["state_loss"]),
                        "sigreg_loss": float(base_stats["sigreg_loss"]),
                        "ftv_loss_raw": float(ftv_component.detach()),
                        "lambda_ftv": LAMBDA_FTV,
                        "stochastic_seed": stochastic_seed,
                        "model_mode": "train_fixed_rng",
                        "deterministic_algorithms": True,
                        "paired_forward_outputs_exact": True,
                        "component_forward_backward_count": 2,
                        "optimizer_created": False,
                        "optimizer_step": False,
                        "pcr_signal_used": False,
                        "model_state_sha256_before": before,
                        "model_state_sha256_after": "PENDING",
                        "source_contract_sha256": file_sha256(SOURCE_CONTRACT),
                        "public_manifest_sha256": public_manifest_sha,
                        "contains_patient_ids": False,
                    }
                )
            del (
                base_grads,
                ftv_grads,
                base_output,
                ftv_output,
                base_component,
                ftv_component,
            )
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    after = checkpoint_state_fingerprint({"state_dict": model.state_dict()})
    if before != after:
        raise ValueError("gradient audit 意外修改 checkpoint parameter/buffer state")
    for row in rows:
        row["model_state_sha256_after"] = after
        if tuple(row) != BATCH_GRADIENT_COLUMNS:
            raise AssertionError("gradient row column order/schema 漂移")
    destination = output_path(seed_base, fold, checkpoint_kind)
    atomic_csv(destination, rows, overwrite=overwrite)
    validate_gradient_file(
        destination, seed_base, fold, checkpoint_kind, checkpoint_sha
    )
    return destination


def validate_gradient_file(
    path: Path,
    seed_base: int,
    fold: int,
    checkpoint_kind: str,
    checkpoint_sha: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_rows = len(SPLITS) * BATCHES_PER_SPLIT * len(GROUPS)
    if tuple(frame.columns) != BATCH_GRADIENT_COLUMNS:
        raise ValueError(f"gradient file column schema 漂移: {path}")
    expected_keys = {
        (split, batch_index, group)
        for split in SPLITS
        for batch_index in range(BATCHES_PER_SPLIT)
        for group in GROUPS
    }
    observed_keys = set(
        frame[["split", "batch_index", "group"]].itertuples(index=False, name=None)
    )
    if (
        len(frame) != expected_rows
        or frame.duplicated(["split", "batch_id", "group"]).any()
        or observed_keys != expected_keys
    ):
        raise ValueError(f"gradient file row/key coverage 错误: {path}")
    if set(frame["seed_base"]) != {seed_base} or set(frame["fold"]) != {fold}:
        raise ValueError("gradient file seed/fold 漂移")
    if set(frame["checkpoint_kind"]) != {checkpoint_kind} or set(frame["group"]) != set(
        GROUPS
    ):
        raise ValueError("gradient file checkpoint/group 漂移")
    if checkpoint_sha is not None and set(frame["checkpoint_sha256"]) != {
        checkpoint_sha
    }:
        raise ValueError("gradient file checkpoint SHA 漂移")
    current_checkpoint = checkpoint_path(
        seed_base, fold, "g3", "best" if checkpoint_kind == "selected" else "last"
    )
    if set(frame["checkpoint_sha256"]) != {file_sha256(current_checkpoint)}:
        raise ValueError("gradient file checkpoint SHA 与当前文件不一致")
    public = pd.read_csv(PUBLIC_MANIFEST)
    public_sha = file_sha256(PUBLIC_MANIFEST)
    if set(frame["public_manifest_sha256"]) != {public_sha}:
        raise ValueError("gradient file public manifest SHA 漂移")
    if set(frame["source_contract_sha256"]) != {file_sha256(SOURCE_CONTRACT)}:
        raise ValueError("gradient file source contract SHA 漂移")
    public_lookup = {
        (int(row.fold), str(row.split), int(row.batch_index)): row
        for row in public.itertuples(index=False)
    }
    numeric = [
        "base_gradient_norm",
        "ftv_gradient_norm_raw",
        "weighted_ftv_gradient_norm",
        "gradient_dot_raw",
        "gradient_cosine",
        "weighted_gradient_norm_ratio",
        "base_descent_margin",
        "ftv_descent_margin",
        "base_objective",
        "state_loss",
        "sigreg_loss",
        "ftv_loss_raw",
    ]
    if (
        frame[numeric].isna().any().any()
        or not np.isfinite(frame[numeric].to_numpy(dtype=float)).all()
    ):
        raise ValueError("gradient file 核心指标 nonfinite")
    if (frame["base_gradient_none_count"] != 0).any() or (
        frame["ftv_gradient_none_count"] != 0
    ).any():
        raise ValueError("gradient file shared gradient 出现 None")
    if not frame["gradient_cosine"].between(-1.0000001, 1.0000001).all():
        raise ValueError("gradient cosine 越界")
    expected_shared = EXPECTED_GROUP_COUNTS
    invariant_columns = [
        "base_gate",
        "base_degradation",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "ordered_members_hmac_sha256",
        "n_total",
        "n_ftv_available",
        "n_ftv_valid_visits",
        "base_objective",
        "state_loss",
        "sigreg_loss",
        "ftv_loss_raw",
        "lambda_ftv",
        "stochastic_seed",
        "model_mode",
        "model_state_sha256_before",
        "model_state_sha256_after",
        "source_contract_sha256",
        "public_manifest_sha256",
    ]
    for (_, split, batch_index), batch in frame.groupby(
        ["fold", "split", "batch_index"], sort=False
    ):
        if len(batch) != len(GROUPS) or any(
            batch[column].nunique(dropna=False) != 1 for column in invariant_columns
        ):
            raise ValueError("gradient batch 跨 group invariant 漂移")
        manifest = public_lookup[(fold, str(split), int(batch_index))]
        first = batch.iloc[0]
        expected_batch_id = str(manifest.batch_id)
        if (
            str(first["batch_id"]) != expected_batch_id
            or str(first["ordered_members_hmac_sha256"])
            != str(manifest.ordered_members_hmac_sha256)
            or int(first["n_ftv_available"]) != int(manifest.n_ftv_available)
            or int(first["n_total"]) != BATCH_SIZE
            or int(first["stochastic_seed"])
            != derive_stochastic_seed(fold, str(split), int(batch_index))
        ):
            raise ValueError("gradient batch 与冻结 manifest/RNG 不一致")
    for row in frame.itertuples(index=False):
        tensors, parameters = expected_shared[str(row.group)]
        if (
            int(row.parameter_tensors) != tensors
            or int(row.parameter_count) != parameters
        ):
            raise ValueError("gradient group parameter count 漂移")
        expected_geometry = gradient_geometry(
            float(row.base_gradient_norm) ** 2,
            float(row.ftv_gradient_norm_raw) ** 2,
            float(row.gradient_dot_raw),
        )
        for field in (
            "weighted_ftv_gradient_norm",
            "gradient_cosine",
            "weighted_gradient_norm_ratio",
            "base_descent_margin",
            "ftv_descent_margin",
        ):
            if not math.isclose(
                float(getattr(row, field)),
                float(expected_geometry[field]),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(f"gradient geometry {field} 不可重算")
        expected_flags = {
            "negative_cosine": float(row.gradient_cosine) < 0,
            "strong_negative_cosine": float(row.gradient_cosine) < -0.1,
            "very_strong_negative_cosine": float(row.gradient_cosine) < -0.25,
            "base_descent_failure": float(row.base_descent_margin) < 0,
        }
        for field, expected in expected_flags.items():
            if _strict_bool(getattr(row, field), field) != expected:
                raise ValueError(f"gradient flag {field} 不可重算")
        required_true = ("deterministic_algorithms", "paired_forward_outputs_exact")
        required_false = (
            "optimizer_created",
            "optimizer_step",
            "pcr_signal_used",
            "contains_patient_ids",
        )
        if not all(
            _strict_bool(getattr(row, field), field) for field in required_true
        ) or any(_strict_bool(getattr(row, field), field) for field in required_false):
            raise ValueError("gradient safety/privacy flags 失败")
        if (
            float(row.lambda_ftv) != LAMBDA_FTV
            or str(row.model_mode) != "train_fixed_rng"
            or int(row.component_forward_backward_count) != 2
            or str(row.model_state_sha256_before) != str(row.model_state_sha256_after)
            or str(row.base_gate)
            != ("PASS" if float(row.base_degradation) <= 0.05 else "FAIL")
        ):
            raise ValueError("gradient execution/state/base-gate contract 失败")
    return frame


def synthetic_self_test(device_name: str = "cpu") -> dict[str, Any]:
    aligned = gradient_geometry(4.0, 9.0, 6.0)
    opposed = gradient_geometry(4.0, 9.0, -6.0)
    orthogonal = gradient_geometry(4.0, 9.0, 0.0)
    model = DGRSWorldModel("G3")
    groups = parameter_groups(model)
    checks = {
        "aligned_cosine_one": math.isclose(aligned["gradient_cosine"], 1.0),
        "opposed_cosine_minus_one": math.isclose(opposed["gradient_cosine"], -1.0),
        "orthogonal_cosine_zero": math.isclose(orthogonal["gradient_cosine"], 0.0),
        "aligned_margin_gt_one": aligned["base_descent_margin"] > 1.0,
        "opposed_margin_lt_one": opposed["base_descent_margin"] < 1.0,
        "zero_norm_is_nan": math.isnan(
            gradient_geometry(0.0, 1.0, 0.0)["gradient_cosine"]
        ),
        "group_names_locked": set(groups) == set(GROUPS),
        "all_shared_count_locked": EXPECTED_GROUP_COUNTS["all_shared"] == (32, 917_824),
    }
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA synthetic self-test 请求但 CUDA 不可用")
    _configure_determinism()
    model = model.to(device).train()
    objective = (
        DGRSObjective("G3", LAMBDA_FTV, 0.09, 256, (2.0, 1.0, 0.5)).to(device).train()
    )
    _set_stochastic_seed(AUDIT_SEED)
    image = torch.randn(2, 4, 7, 8, 16, 16, device=device)
    target = torch.randn(2, 4, device=device)
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    before = checkpoint_state_fingerprint({"state_dict": model.state_dict()})
    model.zero_grad(set_to_none=True)
    _set_stochastic_seed(AUDIT_SEED + 1)
    first = model(image, None)
    _, first_stats = objective(first, target, mask)
    first_stats["_base_component"].backward()
    first_grad = model.response_projection[0].weight.grad
    model.zero_grad(set_to_none=True)
    _set_stochastic_seed(AUDIT_SEED + 1)
    second = model(image, None)
    _, second_stats = objective(second, target, mask)
    checks["paired_fixed_rng_forward_exact"] = _outputs_match(first, second)
    second_stats["_ftv_component_raw"].backward()
    second_grad = model.response_projection[0].weight.grad
    checks["base_and_ftv_shared_gradients_present"] = (
        first_grad is not None and second_grad is not None
    )
    checks["synthetic_state_unchanged"] = before == checkpoint_state_fingerprint(
        {"state_dict": model.state_dict()}
    )
    if not all(checks.values()):
        raise AssertionError(f"gradient self-test 失败: {checks}")
    return {"status": "ok", "checks": checks}


__all__ = [
    "EXPECTED_GROUP_COUNTS",
    "extract_run",
    "gradient_geometry",
    "output_path",
    "parameter_groups",
    "synthetic_self_test",
    "validate_gradient_file",
]
