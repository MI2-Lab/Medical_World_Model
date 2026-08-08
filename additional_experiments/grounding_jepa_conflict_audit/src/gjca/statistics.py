"""Grounding–JEPA audit 的纯统计工具。

本模块不读写实验资产。调用方必须先把 batch 聚合到 seed×fold run
统计单位，并在所有 endpoint 间复用同一份重采样索引。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr


_RESAMPLING_STORAGE_ARRAY_NAMES = (
    "crossed_seed_draws",
    "crossed_fold_draws",
    "crossed_permutation_seed_orders",
    "crossed_permutation_fold_orders",
)


class StatisticsInputError(ValueError):
    """统计函数的 shape、索引或参数合同被违反。"""


class NonFiniteInputError(StatisticsInputError):
    """正式 endpoint 输入包含 NaN 或 Inf；禁止静默 complete-case。"""


@dataclass(frozen=True)
class ResamplingIndices:
    """一次生成、跨 endpoint 同步复用的重采样索引。

    ``crossed_permutation_*_orders`` 穷举全部 row-order×column-order，正式
    exact gate 必须使用它们。bundle 不保存 iid-group 或 unrestricted-run
    permutation 索引。
    """

    crossed_seed_draws: np.ndarray
    crossed_fold_draws: np.ndarray
    crossed_permutation_seed_orders: np.ndarray
    crossed_permutation_fold_orders: np.ndarray
    crossed_seed_rng_seed: int

    def __post_init__(self) -> None:
        names = (
            "crossed_seed_draws",
            "crossed_fold_draws",
            "crossed_permutation_seed_orders",
            "crossed_permutation_fold_orders",
        )
        for name in names:
            frozen = _frozen_integer_matrix(getattr(self, name), name)
            object.__setattr__(self, name, frozen)

        if self.crossed_seed_draws.shape[0] != self.crossed_fold_draws.shape[0]:
            raise StatisticsInputError("crossed seed/fold replicate 数不一致")
        n_seeds = self.crossed_seed_draws.shape[1]
        n_folds = self.crossed_fold_draws.shape[1]
        if min(n_seeds, n_folds) <= 0:
            raise StatisticsInputError("重采样维度必须为正")
        _validate_index_range(self.crossed_seed_draws, n_seeds, "crossed_seed_draws")
        _validate_index_range(self.crossed_fold_draws, n_folds, "crossed_fold_draws")
        _validate_exact_crossed_permutation_orders(
            self.crossed_permutation_seed_orders,
            self.crossed_permutation_fold_orders,
            n_seeds=n_seeds,
            n_folds=n_folds,
        )
        value = self.crossed_seed_rng_seed
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise StatisticsInputError("crossed_seed_rng_seed 必须是非负整数")

    def manifest(self) -> dict[str, object]:
        """返回 .npz 存储数组的 shape/dtype/raw SHA 合同。"""

        arrays = {
            name: _storage_array_manifest(array)
            for name, array in _storage_arrays(self).items()
        }
        payload: dict[str, object] = {
            "schema_version": 2,
            "bit_generator": "PCG64",
            "rng_seeds": {
                "crossed": int(self.crossed_seed_rng_seed),
            },
            "algorithms": {
                "crossed_bootstrap": {
                    "name": "independent_seed_and_fold_draws_with_replacement_then_cartesian_product",
                    "formal_gate_eligible": True,
                    "synchronized_across_all_endpoints": True,
                    "missing_group_policy": "record_nonfinite_never_redraw",
                },
                "crossed_exact_permutation": {
                    "name": "all_seed_orders_cartesian_all_fold_orders",
                    "replicates": int(self.crossed_permutation_seed_orders.shape[0]),
                    "includes_identity": True,
                    "two_sided_tie_rule": "absolute_greater_or_equal",
                    "formal_gate_eligible": True,
                    "synchronized_across_all_endpoints": True,
                },
            },
            "formal_index_arrays": [
                "crossed_seed_draws",
                "crossed_fold_draws",
                "crossed_permutation_seed_orders",
                "crossed_permutation_fold_orders",
            ],
            "excluded_from_bundle": [
                "iid_within_group_bootstrap_indices",
                "unrestricted_run_permutation_indices",
            ],
            "arrays": arrays,
        }
        payload["bundle_sha256"] = _canonical_json_sha256(payload)
        return payload


@dataclass(frozen=True)
class CrossedSpearmanResult:
    """Spearman 点估计、仅敏感性 IID p 与 crossed percentile CI。

    正式 gate 的 p 必须来自 exact_crossed_spearman_permutation；这里的
    scipy IID p 只保留作非正式 sensitivity 描述。
    """

    estimate: float | None
    iid_p_sensitivity_only_not_formal_gate: float | None
    ci_low: float | None
    ci_high: float | None
    bootstrap_requested: int
    bootstrap_finite: int
    bootstrap_finite_fraction: float
    confidence_level: float
    status: str

    @property
    def p_raw(self) -> float | None:
        """兼容别名；仅 IID sensitivity，禁止用于正式 gate。"""

        return self.iid_p_sensitivity_only_not_formal_gate


@dataclass(frozen=True)
class BootstrapEstimate:
    """一个点估计及其 percentile bootstrap CI。"""

    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    bootstrap_requested: int
    bootstrap_finite: int
    bootstrap_finite_fraction: float
    confidence_level: float
    status: str


@dataclass(frozen=True)
class GroupBootstrapResult:
    """FAIL−PASS mean/median difference 与 FAIL/PASS median ratio。"""

    mean_difference: BootstrapEstimate
    median_difference: BootstrapEstimate
    median_ratio: BootstrapEstimate


@dataclass(frozen=True)
class PermutationResult:
    """旧版双侧 run-label Monte Carlo permutation；不得用于正式 gate。"""

    estimate: float
    p_value: float
    replicates: int
    extreme_count: int
    contrast: str
    status: str = "compatibility_only_not_formal_gate"


@dataclass(frozen=True)
class ExactCrossedPermutationResult:
    """穷举 crossed row/column orders 的双侧 exact permutation 结果。"""

    estimate: float | None
    p_value: float | None
    replicates: int
    extreme_count: int
    statistic: str
    includes_identity: bool
    status: str


@dataclass(frozen=True)
class HolmEndpoint:
    """一个预声明 family member 的 Holm step-down 结果。"""

    endpoint_id: str
    p_raw: float | None
    p_effective: float
    p_holm: float
    rank: int
    family_size: int
    status: str


def generate_resampling_indices(
    *,
    crossed_replicates: int,
    n_seeds: int = 5,
    n_folds: int = 5,
    crossed_seed: int = 2026080801,
) -> ResamplingIndices:
    """生成 crossed bootstrap 与完整 exact crossed permutation 索引。

    函数不接收 endpoint，因此返回的索引必须由调用方在所有统计量间
    同步复用。
    """

    counts = {
        "crossed_replicates": crossed_replicates,
        "n_seeds": n_seeds,
        "n_folds": n_folds,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise StatisticsInputError(f"{name} 必须是正整数")
    if max(int(n_seeds), int(n_folds)) > 256:
        raise StatisticsInputError("uint8 索引存储要求各 level 不超过 256")
    exact_replicates = math.factorial(int(n_seeds)) * math.factorial(int(n_folds))
    if exact_replicates > 2_000_000:
        raise StatisticsInputError("exact crossed permutation 组合数超过安全上限")
    if (
        isinstance(crossed_seed, bool)
        or int(crossed_seed) != crossed_seed
        or int(crossed_seed) < 0
    ):
        raise StatisticsInputError("crossed_seed 必须是非负整数")

    crossed_rng = np.random.Generator(np.random.PCG64(int(crossed_seed)))
    seed_draws = crossed_rng.integers(
        0, int(n_seeds), size=(int(crossed_replicates), int(n_seeds)), dtype=np.int64
    )
    fold_draws = crossed_rng.integers(
        0, int(n_folds), size=(int(crossed_replicates), int(n_folds)), dtype=np.int64
    )
    exact_seed_orders, exact_fold_orders = _all_crossed_permutation_orders(
        int(n_seeds), int(n_folds)
    )
    return ResamplingIndices(
        crossed_seed_draws=seed_draws,
        crossed_fold_draws=fold_draws,
        crossed_permutation_seed_orders=exact_seed_orders,
        crossed_permutation_fold_orders=exact_fold_orders,
        crossed_seed_rng_seed=int(crossed_seed),
    )


def index_array_sha256(array: np.ndarray) -> str:
    """对整数索引的 canonical little-endian int64 shape+bytes 取 SHA-256。"""

    normalized = _canonical_integer_array(array, "index_array")
    header = json.dumps(
        {"dtype": "<i8", "shape": list(normalized.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def array_raw_sha256(array: np.ndarray) -> str:
    """对数组按当前 dtype 的 C-order raw bytes 取 SHA-256。

    shape 与 dtype 在 manifest 中单独冻结；本函数只对 raw payload 取摘要。
    """

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise StatisticsInputError("raw SHA 禁止 object dtype")
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def save_resampling_indices_npz(
    path: str | Path,
    indices: ResamplingIndices,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """原子写入压缩 uint8 `.npz` bundle 并返回 manifest。

    文件内同时保存 canonical JSON manifest；返回值额外包含
    ``npz_sha256``，但该文件级 SHA 不参与内部 bundle SHA 以避免自引用。
    """

    if not isinstance(indices, ResamplingIndices):
        raise StatisticsInputError("indices 必须是 ResamplingIndices")
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise StatisticsInputError("重采样 bundle 文件必须使用 .npz 后缀")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有重采样 bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _storage_arrays(indices)
    manifest = indices.manifest()
    manifest_bytes = _canonical_json_bytes(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                **arrays,
                manifest_json=np.frombuffer(manifest_bytes, dtype=np.uint8).copy(),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _manifest_with_file_sha(manifest, destination)


def load_resampling_indices_npz(
    path: str | Path,
) -> tuple[ResamplingIndices, dict[str, object]]:
    """严格校验并加载 `.npz` bundle。

    校验包括 exact member set、uint8 dtype、shape、每数组 raw SHA、
    canonical index SHA、bundle SHA 以及完整 crossed order 合同。
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"重采样 bundle 不存在: {source}")
    expected_array_names = set(_RESAMPLING_STORAGE_ARRAY_NAMES)
    expected_members = expected_array_names | {"manifest_json"}
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != expected_members:
                raise StatisticsInputError(
                    f"npz member set 不一致: {sorted(archive.files)}"
                )
            manifest_raw = np.asarray(archive["manifest_json"])
            if manifest_raw.dtype != np.uint8 or manifest_raw.ndim != 1:
                raise StatisticsInputError("manifest_json 必须是 1D uint8")
            try:
                manifest = json.loads(manifest_raw.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StatisticsInputError(
                    "manifest_json 不是合法 canonical JSON"
                ) from error
            if not isinstance(manifest, dict):
                raise StatisticsInputError("manifest_json root 必须是 object")
            arrays = {
                name: np.ascontiguousarray(archive[name])
                for name in _RESAMPLING_STORAGE_ARRAY_NAMES
            }
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        if isinstance(error, StatisticsInputError):
            raise
        raise StatisticsInputError(f"无法读取重采样 npz: {source}") from error

    _validate_stored_manifest(manifest, arrays)
    rng_seeds = manifest.get("rng_seeds")
    if not isinstance(rng_seeds, dict):
        raise StatisticsInputError("manifest 缺 rng_seeds")
    loaded = ResamplingIndices(
        crossed_seed_draws=arrays["crossed_seed_draws"],
        crossed_fold_draws=arrays["crossed_fold_draws"],
        crossed_permutation_seed_orders=arrays["crossed_permutation_seed_orders"],
        crossed_permutation_fold_orders=arrays["crossed_permutation_fold_orders"],
        crossed_seed_rng_seed=_strict_manifest_integer(rng_seeds, "crossed"),
    )
    if loaded.manifest() != manifest:
        raise StatisticsInputError("加载后重算 manifest 与文件内 manifest 不一致")
    return loaded, _manifest_with_file_sha(manifest, source)


def spearman_crossed_ci(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    seed_draws: np.ndarray,
    fold_draws: np.ndarray,
    *,
    confidence_level: float = 0.95,
    minimum_finite_fraction: float = 0.95,
) -> CrossedSpearmanResult:
    """在完整 seed×fold grid 上计算 Spearman 与 crossed percentile CI。

    每个 replicate 用同行的 seed/fold draws 构造 Cartesian product；重复
    cell 按 multiplicity 展开后重新 average-rank。常量点估计返回
    ``constant_input``，而 NaN/Inf 输入直接拒绝。返回的
    ``iid_p_sensitivity_only_not_formal_gate`` 是 scipy IID sensitivity；
    正式 gate 必须另用 exact crossed permutation p。
    """

    x = _strict_finite_float_array(x_grid, "x_grid", ndim=2)
    y = _strict_finite_float_array(y_grid, "y_grid", ndim=2)
    if x.shape != y.shape:
        raise StatisticsInputError("x_grid/y_grid shape 不一致")
    seeds = _integer_matrix(seed_draws, "seed_draws")
    folds = _integer_matrix(fold_draws, "fold_draws")
    if seeds.shape[0] != folds.shape[0]:
        raise StatisticsInputError("seed/fold bootstrap replicate 数不一致")
    if seeds.shape[1] != x.shape[0] or folds.shape[1] != x.shape[1]:
        raise StatisticsInputError("crossed draws 每行必须分别抽 n_seed/n_fold 次")
    _validate_index_range(seeds, x.shape[0], "seed_draws")
    _validate_index_range(folds, x.shape[1], "fold_draws")
    confidence_level, minimum_finite_fraction = _validate_interval_policy(
        confidence_level, minimum_finite_fraction
    )

    requested = seeds.shape[0]
    if _is_constant(x.ravel()) or _is_constant(y.ravel()):
        return CrossedSpearmanResult(
            estimate=None,
            iid_p_sensitivity_only_not_formal_gate=None,
            ci_low=None,
            ci_high=None,
            bootstrap_requested=requested,
            bootstrap_finite=0,
            bootstrap_finite_fraction=0.0,
            confidence_level=confidence_level,
            status="constant_input",
        )

    point = spearmanr(x.ravel(), y.ravel(), alternative="two-sided")
    estimate = float(point.statistic)
    iid_p_sensitivity = float(point.pvalue)
    if not math.isfinite(estimate) or not math.isfinite(iid_p_sensitivity):
        raise StatisticsInputError("非常量输入产生 nonfinite Spearman")

    bootstrap = np.full(requested, np.nan, dtype=np.float64)
    for replicate in range(requested):
        sample_x = x[np.ix_(seeds[replicate], folds[replicate])].ravel()
        sample_y = y[np.ix_(seeds[replicate], folds[replicate])].ravel()
        if _is_constant(sample_x) or _is_constant(sample_y):
            continue
        value = float(spearmanr(sample_x, sample_y).statistic)
        if math.isfinite(value):
            bootstrap[replicate] = value
    interval = _percentile_result(
        estimate,
        bootstrap,
        confidence_level=confidence_level,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    return CrossedSpearmanResult(
        estimate=estimate,
        iid_p_sensitivity_only_not_formal_gate=iid_p_sensitivity,
        ci_low=interval.ci_low,
        ci_high=interval.ci_high,
        bootstrap_requested=interval.bootstrap_requested,
        bootstrap_finite=interval.bootstrap_finite,
        bootstrap_finite_fraction=interval.bootstrap_finite_fraction,
        confidence_level=confidence_level,
        status=interval.status,
    )


def exact_crossed_spearman_permutation(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    seed_orders: np.ndarray,
    fold_orders: np.ndarray,
) -> ExactCrossedPermutationResult:
    """穷举 row/column orders，计算 Spearman 的双侧 exact p。

    ``x_grid`` 固定；每个 replicate 将 ``y_grid`` 同步按一个 seed order 与
    一个 fold order 重排。orders 必须是全部 ``n_seed!×n_fold!`` 组合且
    包含 identity。p 为 ``count(|T_perm| >= |T_obs|) / B``，不使用 plus-one。
    """

    x = _strict_finite_float_array(x_grid, "x_grid", ndim=2)
    y = _strict_finite_float_array(y_grid, "y_grid", ndim=2)
    if x.shape != y.shape:
        raise StatisticsInputError("x_grid/y_grid shape 不一致")
    seeds, folds = _validated_exact_orders(seed_orders, fold_orders, x.shape)
    replicates = seeds.shape[0]
    if _is_constant(x.ravel()) or _is_constant(y.ravel()):
        return ExactCrossedPermutationResult(
            estimate=None,
            p_value=None,
            replicates=replicates,
            extreme_count=0,
            statistic="spearman",
            includes_identity=True,
            status="constant_input",
        )
    # Spearman rho 是 average ranks 的 Pearson correlation。permutation 不改变
    # y 的 rank multiset，因此只需 rank 两次，再批量移动 centered y ranks。
    x_ranks = rankdata(x.ravel(), method="average").reshape(x.shape)
    y_ranks = rankdata(y.ravel(), method="average").reshape(y.shape)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = math.sqrt(
        float(np.square(x_centered).sum()) * float(np.square(y_centered).sum())
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise StatisticsInputError("非常量输入产生非法 Spearman denominator")
    permuted_y = y_centered[seeds[:, :, None], folds[:, None, :]]
    permutation_values = (
        np.einsum("bij,ij->b", permuted_y, x_centered, optimize=True) / denominator
    )
    if not np.isfinite(permutation_values).all():
        raise StatisticsInputError("exact crossed permutation 产生 nonfinite Spearman")
    # 从已验证的唯一 identity 行取 observed，避免单独浮点路径将 identity
    # 误判为不足阈值。
    observed = float(permutation_values[_identity_order_index(seeds, folds)])
    extreme = int(np.count_nonzero(np.abs(permutation_values) >= abs(observed)))
    return ExactCrossedPermutationResult(
        estimate=observed,
        p_value=float(extreme / replicates),
        replicates=replicates,
        extreme_count=extreme,
        statistic="spearman",
        includes_identity=True,
        status="ok",
    )


def exact_crossed_group_permutation(
    values_grid: np.ndarray,
    pass_mask_grid: np.ndarray,
    seed_orders: np.ndarray,
    fold_orders: np.ndarray,
    *,
    contrast: str,
) -> ExactCrossedPermutationResult:
    """在固定 PASS-mask grid 上穷举 outcome row/column orders。

    统计量固定为 FAIL−PASS mean 或 median difference；mask 不移动，仅 outcome
    matrix 按完整 crossed orders 移动。使用含 identity 的 exact 双侧 p。
    """

    values = _strict_finite_float_array(values_grid, "values_grid", ndim=2)
    mask = _strict_bool_grid(pass_mask_grid, "pass_mask_grid", values.shape)
    if contrast not in {"mean", "median"}:
        raise StatisticsInputError("contrast 必须是 'mean' 或 'median'")
    if not mask.any() or mask.all():
        raise StatisticsInputError("PASS/FAIL 两组都必须非空")
    seeds, folds = _validated_exact_orders(seed_orders, fold_orders, values.shape)
    replicates = seeds.shape[0]
    permuted = values[seeds[:, :, None], folds[:, None, :]].reshape(
        replicates, values.size
    )
    pass_positions = mask.ravel()
    if contrast == "mean":
        permutation_values = permuted[:, ~pass_positions].mean(axis=1) - permuted[
            :, pass_positions
        ].mean(axis=1)
    else:
        permutation_values = np.median(
            permuted[:, ~pass_positions], axis=1
        ) - np.median(permuted[:, pass_positions], axis=1)
    if not np.isfinite(permutation_values).all():
        raise StatisticsInputError("finite grid 产生 nonfinite exact group statistic")
    observed = float(permutation_values[_identity_order_index(seeds, folds)])
    extreme = int(np.count_nonzero(np.abs(permutation_values) >= abs(observed)))
    return ExactCrossedPermutationResult(
        estimate=observed,
        p_value=float(extreme / replicates),
        replicates=replicates,
        extreme_count=extreme,
        statistic=f"fail_minus_pass_{contrast}",
        includes_identity=True,
        status="ok",
    )


def crossed_group_bootstrap_contrasts(
    values_grid: np.ndarray,
    pass_mask_grid: np.ndarray,
    seed_draws: np.ndarray,
    fold_draws: np.ndarray,
    *,
    confidence_level: float = 0.95,
    minimum_finite_fraction: float = 0.95,
) -> GroupBootstrapResult:
    """crossed seed×fold bootstrap 的 PASS/FAIL contrasts。

    每个 replicate 对 outcome 与 PASS mask 同步抽取 Cartesian grid，重复 cell
    按 multiplicity 展开。若重采样后缺任一组，则三个统计量均记 nonfinite；
    median PASS denominator 非正时 ratio 单独记 nonfinite。任何 invalid replicate
    都不会重抽。
    """

    values = _strict_finite_float_array(values_grid, "values_grid", ndim=2)
    mask = _strict_bool_grid(pass_mask_grid, "pass_mask_grid", values.shape)
    if not mask.any() or mask.all():
        raise StatisticsInputError("PASS/FAIL 两组都必须非空")
    seeds = _integer_matrix(seed_draws, "seed_draws")
    folds = _integer_matrix(fold_draws, "fold_draws")
    if seeds.shape[0] != folds.shape[0]:
        raise StatisticsInputError("seed/fold bootstrap replicate 数不一致")
    if seeds.shape[1] != values.shape[0] or folds.shape[1] != values.shape[1]:
        raise StatisticsInputError("crossed draws 每行必须分别抽 n_seed/n_fold 次")
    _validate_index_range(seeds, values.shape[0], "seed_draws")
    _validate_index_range(folds, values.shape[1], "fold_draws")
    confidence_level, minimum_finite_fraction = _validate_interval_policy(
        confidence_level, minimum_finite_fraction
    )

    pass_values = values[mask]
    fail_values = values[~mask]
    pass_point_median = float(np.median(pass_values))
    fail_point_median = float(np.median(fail_values))
    mean_point = float(fail_values.mean() - pass_values.mean())
    median_point = fail_point_median - pass_point_median
    requested = seeds.shape[0]
    mean_bootstrap = np.full(requested, np.nan, dtype=np.float64)
    median_bootstrap = np.full(requested, np.nan, dtype=np.float64)
    ratio_bootstrap = np.full(requested, np.nan, dtype=np.float64)
    for replicate in range(requested):
        sampled_values = values[np.ix_(seeds[replicate], folds[replicate])]
        sampled_mask = mask[np.ix_(seeds[replicate], folds[replicate])]
        if not sampled_mask.any() or sampled_mask.all():
            continue
        sampled_pass = sampled_values[sampled_mask]
        sampled_fail = sampled_values[~sampled_mask]
        pass_median = float(np.median(sampled_pass))
        fail_median = float(np.median(sampled_fail))
        mean_bootstrap[replicate] = sampled_fail.mean() - sampled_pass.mean()
        median_bootstrap[replicate] = fail_median - pass_median
        if pass_median > 0:
            ratio_bootstrap[replicate] = fail_median / pass_median

    mean_result = _percentile_result(
        mean_point,
        mean_bootstrap,
        confidence_level=confidence_level,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    median_result = _percentile_result(
        median_point,
        median_bootstrap,
        confidence_level=confidence_level,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    if pass_point_median <= 0:
        ratio_result = BootstrapEstimate(
            estimate=None,
            ci_low=None,
            ci_high=None,
            bootstrap_requested=requested,
            bootstrap_finite=0,
            bootstrap_finite_fraction=0.0,
            confidence_level=confidence_level,
            status="invalid_ratio_denominator",
        )
    else:
        ratio_result = _percentile_result(
            fail_point_median / pass_point_median,
            ratio_bootstrap,
            confidence_level=confidence_level,
            minimum_finite_fraction=minimum_finite_fraction,
        )
    return GroupBootstrapResult(
        mean_difference=mean_result,
        median_difference=median_result,
        median_ratio=ratio_result,
    )


def group_bootstrap_contrasts(
    values: np.ndarray,
    pass_mask: np.ndarray,
    pass_draws: np.ndarray,
    fail_draws: np.ndarray,
    *,
    confidence_level: float = 0.95,
    minimum_finite_fraction: float = 0.95,
) -> GroupBootstrapResult:
    """兼容版 iid 组内 bootstrap；不得用于正式 crossed gate。

    ``pass_draws``/``fail_draws`` 索引各自按 run 顺序排列的局部组数组，
    而非原 25-run 数组的绝对位置。
    """

    vector = _strict_finite_float_array(values, "values", ndim=1)
    mask = _strict_bool_vector(pass_mask, "pass_mask", vector.size)
    pass_values = vector[mask]
    fail_values = vector[~mask]
    if pass_values.size == 0 or fail_values.size == 0:
        raise StatisticsInputError("PASS/FAIL 两组都必须非空")
    pass_indices = _integer_matrix(pass_draws, "pass_draws")
    fail_indices = _integer_matrix(fail_draws, "fail_draws")
    if pass_indices.shape != (fail_indices.shape[0], pass_values.size):
        if pass_indices.shape[0] != fail_indices.shape[0]:
            raise StatisticsInputError("PASS/FAIL group bootstrap replicate 数不一致")
        if pass_indices.shape[1] != pass_values.size:
            raise StatisticsInputError("pass_draws 每行必须抽 n_pass 次")
    if fail_indices.shape[1] != fail_values.size:
        raise StatisticsInputError("fail_draws 每行必须抽 n_fail 次")
    _validate_index_range(pass_indices, pass_values.size, "pass_draws")
    _validate_index_range(fail_indices, fail_values.size, "fail_draws")
    confidence_level, minimum_finite_fraction = _validate_interval_policy(
        confidence_level, minimum_finite_fraction
    )

    pass_samples = pass_values[pass_indices]
    fail_samples = fail_values[fail_indices]
    mean_bootstrap = fail_samples.mean(axis=1) - pass_samples.mean(axis=1)
    pass_medians = np.median(pass_samples, axis=1)
    fail_medians = np.median(fail_samples, axis=1)
    median_bootstrap = fail_medians - pass_medians

    mean_point = float(fail_values.mean() - pass_values.mean())
    pass_point_median = float(np.median(pass_values))
    fail_point_median = float(np.median(fail_values))
    median_point = fail_point_median - pass_point_median
    mean_result = _percentile_result(
        mean_point,
        mean_bootstrap,
        confidence_level=confidence_level,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    median_result = _percentile_result(
        median_point,
        median_bootstrap,
        confidence_level=confidence_level,
        minimum_finite_fraction=minimum_finite_fraction,
    )

    if pass_point_median <= 0:
        ratio_result = BootstrapEstimate(
            estimate=None,
            ci_low=None,
            ci_high=None,
            bootstrap_requested=pass_indices.shape[0],
            bootstrap_finite=0,
            bootstrap_finite_fraction=0.0,
            confidence_level=confidence_level,
            status="invalid_ratio_denominator",
        )
    else:
        ratio_bootstrap = np.full(pass_medians.shape, np.nan, dtype=np.float64)
        valid = pass_medians > 0
        ratio_bootstrap[valid] = fail_medians[valid] / pass_medians[valid]
        ratio_result = _percentile_result(
            fail_point_median / pass_point_median,
            ratio_bootstrap,
            confidence_level=confidence_level,
            minimum_finite_fraction=minimum_finite_fraction,
        )
    return GroupBootstrapResult(
        mean_difference=mean_result,
        median_difference=median_result,
        median_ratio=ratio_result,
    )


def two_sided_permutation_test(
    values: np.ndarray,
    pass_mask: np.ndarray,
    permutation_orders: np.ndarray,
    *,
    contrast: str,
) -> PermutationResult:
    """兼容版 unrestricted 25-run plus-one permutation；不得用于正式 gate。"""

    vector = _strict_finite_float_array(values, "values", ndim=1)
    mask = _strict_bool_vector(pass_mask, "pass_mask", vector.size)
    n_pass = int(mask.sum())
    n_fail = int((~mask).sum())
    if n_pass == 0 or n_fail == 0:
        raise StatisticsInputError("PASS/FAIL 两组都必须非空")
    orders = _integer_matrix(permutation_orders, "permutation_orders")
    if orders.shape[1] != vector.size:
        raise StatisticsInputError("permutation_orders 宽度与 run 数不一致")
    _validate_permutation_orders(orders)
    if contrast not in {"mean", "median"}:
        raise StatisticsInputError("contrast 必须是 'mean' 或 'median'")

    observed = _difference(vector[~mask], vector[mask], contrast)
    pseudo_fail = vector[orders[:, :n_fail]]
    pseudo_pass = vector[orders[:, n_fail:]]
    if contrast == "mean":
        permutation_values = pseudo_fail.mean(axis=1) - pseudo_pass.mean(axis=1)
    else:
        permutation_values = np.median(pseudo_fail, axis=1) - np.median(
            pseudo_pass, axis=1
        )
    if not np.isfinite(permutation_values).all():
        raise StatisticsInputError("finite values 产生 nonfinite permutation statistic")
    extreme = int(np.count_nonzero(np.abs(permutation_values) >= abs(observed)))
    replicates = int(orders.shape[0])
    return PermutationResult(
        estimate=observed,
        p_value=float((1 + extreme) / (replicates + 1)),
        replicates=replicates,
        extreme_count=extreme,
        contrast=contrast,
    )


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, HolmEndpoint]:
    """对固定 family 执行 Holm step-down；NA/nonfinite p 按 1 但不删除。"""

    if not p_values:
        raise StatisticsInputError("Holm family 不得为空")
    prepared: list[tuple[str, float | None, float, str]] = []
    for endpoint_id, raw in p_values.items():
        if not isinstance(endpoint_id, str) or not endpoint_id.strip():
            raise StatisticsInputError("Holm endpoint id 必须是非空字符串")
        if raw is None:
            prepared.append((endpoint_id, None, 1.0, "unavailable_substituted_one"))
            continue
        if isinstance(raw, bool):
            raise StatisticsInputError(f"{endpoint_id} p 不得是 bool")
        value = float(raw)
        if not math.isfinite(value):
            prepared.append((endpoint_id, None, 1.0, "unavailable_substituted_one"))
        elif not 0.0 <= value <= 1.0:
            raise StatisticsInputError(f"{endpoint_id} p 必须在 [0,1]")
        else:
            prepared.append((endpoint_id, value, value, "ok"))

    ordered = sorted(prepared, key=lambda item: (item[2], item[0]))
    family_size = len(ordered)
    adjusted_by_id: dict[str, HolmEndpoint] = {}
    running = 0.0
    for zero_rank, (endpoint_id, raw, effective, status) in enumerate(ordered):
        candidate = min(1.0, (family_size - zero_rank) * effective)
        running = min(1.0, max(running, candidate))
        adjusted_by_id[endpoint_id] = HolmEndpoint(
            endpoint_id=endpoint_id,
            p_raw=raw,
            p_effective=effective,
            p_holm=running,
            rank=zero_rank + 1,
            family_size=family_size,
            status=status,
        )
    return {
        endpoint_id: adjusted_by_id[endpoint_id] for endpoint_id in sorted(p_values)
    }


def _frozen_integer_matrix(array: np.ndarray, name: str) -> np.ndarray:
    matrix = _integer_matrix(array, name).copy(order="C")
    matrix.setflags(write=False)
    return matrix


def _integer_matrix(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise StatisticsInputError(f"{name} 必须是非空二维整数数组")
    if value.dtype.kind not in {"i", "u"}:
        raise StatisticsInputError(f"{name} 必须是整数数组")
    return np.ascontiguousarray(value, dtype=np.int64)


def _canonical_integer_array(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in {"i", "u"}:
        raise StatisticsInputError(f"{name} 必须是整数数组")
    return np.ascontiguousarray(value, dtype="<i8")


def _validate_index_range(indices: np.ndarray, upper: int, name: str) -> None:
    if upper <= 0 or int(indices.min()) < 0 or int(indices.max()) >= upper:
        raise StatisticsInputError(f"{name} 越界；允许范围 [0,{upper})")


def _validate_permutation_orders(orders: np.ndarray) -> None:
    expected = np.arange(orders.shape[1], dtype=np.int64)
    if not np.array_equal(
        np.sort(orders, axis=1), np.broadcast_to(expected, orders.shape)
    ):
        raise StatisticsInputError("permutation_orders 每行必须是 0..n_run-1 的排列")


def _all_crossed_permutation_orders(
    n_seeds: int, n_folds: int
) -> tuple[np.ndarray, np.ndarray]:
    """按 lexicographic seed-major 顺序生成全部 row-order×column-order。"""

    if n_seeds <= 0 or n_folds <= 0:
        raise StatisticsInputError("crossed permutation level 数必须为正")
    seed_permutations = np.asarray(
        list(itertools.permutations(range(n_seeds))), dtype=np.int64
    )
    fold_permutations = np.asarray(
        list(itertools.permutations(range(n_folds))), dtype=np.int64
    )
    seed_orders = np.repeat(seed_permutations, fold_permutations.shape[0], axis=0)
    fold_orders = np.tile(fold_permutations, (seed_permutations.shape[0], 1))
    return seed_orders, fold_orders


def _validate_exact_crossed_permutation_orders(
    seed_orders: np.ndarray,
    fold_orders: np.ndarray,
    *,
    n_seeds: int,
    n_folds: int,
) -> None:
    """拒绝缺失、重复或非 permutation 的 crossed exact order bundle。"""

    seeds = _integer_matrix(seed_orders, "crossed_permutation_seed_orders")
    folds = _integer_matrix(fold_orders, "crossed_permutation_fold_orders")
    expected_replicates = math.factorial(n_seeds) * math.factorial(n_folds)
    if seeds.shape != (expected_replicates, n_seeds) or folds.shape != (
        expected_replicates,
        n_folds,
    ):
        raise StatisticsInputError(
            "crossed exact orders shape 必须为 n_seed!×n_fold! 的完整组合"
        )
    _validate_permutation_orders(seeds)
    _validate_permutation_orders(folds)
    paired = np.concatenate((seeds, folds), axis=1)
    if np.unique(paired, axis=0).shape[0] != expected_replicates:
        raise StatisticsInputError("crossed exact orders 含重复或缺失组合")
    identity_seed = np.arange(n_seeds, dtype=np.int64)
    identity_fold = np.arange(n_folds, dtype=np.int64)
    identity = np.all(seeds == identity_seed, axis=1) & np.all(
        folds == identity_fold, axis=1
    )
    if int(identity.sum()) != 1:
        raise StatisticsInputError("crossed exact orders 必须且只能包含一次 identity")


def _validated_exact_orders(
    seed_orders: np.ndarray,
    fold_orders: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    seeds = _integer_matrix(seed_orders, "seed_orders")
    folds = _integer_matrix(fold_orders, "fold_orders")
    _validate_exact_crossed_permutation_orders(
        seeds,
        folds,
        n_seeds=grid_shape[0],
        n_folds=grid_shape[1],
    )
    return seeds, folds


def _identity_order_index(seed_orders: np.ndarray, fold_orders: np.ndarray) -> int:
    """返回完整 crossed orders 中唯一 identity 的位置。"""

    identity = np.all(
        seed_orders == np.arange(seed_orders.shape[1], dtype=np.int64), axis=1
    ) & np.all(fold_orders == np.arange(fold_orders.shape[1], dtype=np.int64), axis=1)
    positions = np.flatnonzero(identity)
    if positions.size != 1:
        raise StatisticsInputError("crossed exact orders 必须且只能包含一次 identity")
    return int(positions[0])


def _storage_arrays(indices: ResamplingIndices) -> dict[str, np.ndarray]:
    source = {
        "crossed_seed_draws": indices.crossed_seed_draws,
        "crossed_fold_draws": indices.crossed_fold_draws,
        "crossed_permutation_seed_orders": indices.crossed_permutation_seed_orders,
        "crossed_permutation_fold_orders": indices.crossed_permutation_fold_orders,
    }
    stored: dict[str, np.ndarray] = {}
    for name in _RESAMPLING_STORAGE_ARRAY_NAMES:
        array = _integer_matrix(source[name], name)
        if int(array.min()) < 0 or int(array.max()) > np.iinfo(np.uint8).max:
            raise StatisticsInputError(f"{name} 不能无损存为 uint8")
        stored[name] = np.ascontiguousarray(array, dtype=np.uint8)
    return stored


def _storage_array_manifest(array: np.ndarray) -> dict[str, object]:
    stored = np.asarray(array)
    if stored.dtype != np.uint8 or stored.ndim != 2:
        raise StatisticsInputError("storage manifest 只接受 2D uint8 数组")
    return {
        "shape": list(stored.shape),
        "dtype": "|u1",
        "raw_sha256": array_raw_sha256(stored),
        "canonical_index_sha256": index_array_sha256(stored),
    }


def _validate_stored_manifest(
    manifest: dict[str, object], arrays: Mapping[str, np.ndarray]
) -> None:
    if manifest.get("schema_version") != 2 or manifest.get("bit_generator") != "PCG64":
        raise StatisticsInputError("resampling manifest schema/bit generator 非法")
    claimed_bundle_sha = manifest.get("bundle_sha256")
    if not isinstance(claimed_bundle_sha, str) or len(claimed_bundle_sha) != 64:
        raise StatisticsInputError("resampling manifest 缺合法 bundle SHA")
    unsigned = dict(manifest)
    del unsigned["bundle_sha256"]
    if _canonical_json_sha256(unsigned) != claimed_bundle_sha:
        raise StatisticsInputError("resampling manifest bundle SHA 不匹配")
    declared_arrays = manifest.get("arrays")
    if not isinstance(declared_arrays, dict) or set(declared_arrays) != set(
        _RESAMPLING_STORAGE_ARRAY_NAMES
    ):
        raise StatisticsInputError("resampling manifest array member set 不一致")
    if set(arrays) != set(_RESAMPLING_STORAGE_ARRAY_NAMES):
        raise StatisticsInputError("loaded array member set 不一致")
    for name in _RESAMPLING_STORAGE_ARRAY_NAMES:
        array = np.asarray(arrays[name])
        if array.dtype != np.uint8 or array.ndim != 2:
            raise StatisticsInputError(f"{name} 必须是 2D uint8")
        expected = _storage_array_manifest(array)
        if declared_arrays.get(name) != expected:
            raise StatisticsInputError(
                f"{name} shape/dtype/raw SHA/canonical SHA 不匹配"
            )


def _strict_manifest_integer(payload: Mapping[str, object], key: str) -> int:
    if key not in payload:
        raise StatisticsInputError(f"manifest 缺整数字段: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StatisticsInputError(f"manifest {key} 必须是非负整数")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_with_file_sha(
    manifest: Mapping[str, object], path: Path
) -> dict[str, object]:
    payload = dict(manifest)
    payload["npz_sha256"] = _file_sha256(path)
    payload["npz_bytes"] = path.stat().st_size
    return payload


def _strict_finite_float_array(
    array: np.ndarray, name: str, *, ndim: int
) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != ndim or value.size == 0:
        raise StatisticsInputError(f"{name} 必须是非空 {ndim}D 数组")
    if not np.isfinite(value).all():
        raise NonFiniteInputError(f"{name} 包含 NaN/Inf")
    return value


def _strict_bool_vector(array: np.ndarray, name: str, size: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1 or value.size != size or value.dtype.kind != "b":
        raise StatisticsInputError(f"{name} 必须是长度 {size} 的 bool 数组")
    return value


def _strict_bool_grid(
    array: np.ndarray, name: str, shape: tuple[int, int]
) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or value.ndim != 2 or value.dtype.kind != "b":
        raise StatisticsInputError(f"{name} 必须是 shape={shape} 的 bool grid")
    return value


def _validate_interval_policy(
    confidence_level: float, minimum_finite_fraction: float
) -> tuple[float, float]:
    confidence = float(confidence_level)
    minimum = float(minimum_finite_fraction)
    if not 0.0 < confidence < 1.0:
        raise StatisticsInputError("confidence_level 必须在 (0,1)")
    if not 0.0 < minimum <= 1.0:
        raise StatisticsInputError("minimum_finite_fraction 必须在 (0,1]")
    return confidence, minimum


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.all(values == values[0]))


def _percentile_result(
    estimate: float,
    bootstrap: np.ndarray,
    *,
    confidence_level: float,
    minimum_finite_fraction: float,
) -> BootstrapEstimate:
    samples = np.asarray(bootstrap, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise StatisticsInputError("bootstrap 必须是非空 1D 数组")
    if not math.isfinite(float(estimate)):
        raise StatisticsInputError("point estimate 必须 finite")
    finite = samples[np.isfinite(samples)]
    requested = int(samples.size)
    count = int(finite.size)
    fraction = float(count / requested)
    if fraction < minimum_finite_fraction:
        return BootstrapEstimate(
            estimate=float(estimate),
            ci_low=None,
            ci_high=None,
            bootstrap_requested=requested,
            bootstrap_finite=count,
            bootstrap_finite_fraction=fraction,
            confidence_level=confidence_level,
            status="insufficient_finite_bootstrap",
        )
    alpha = 1.0 - confidence_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return BootstrapEstimate(
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        bootstrap_requested=requested,
        bootstrap_finite=count,
        bootstrap_finite_fraction=fraction,
        confidence_level=confidence_level,
        status="ok",
    )


def _difference(fail: np.ndarray, passed: np.ndarray, contrast: str) -> float:
    if contrast == "mean":
        return float(fail.mean() - passed.mean())
    if contrast == "median":
        return float(np.median(fail) - np.median(passed))
    raise StatisticsInputError("contrast 必须是 'mean' 或 'median'")


__all__ = [
    "BootstrapEstimate",
    "CrossedSpearmanResult",
    "ExactCrossedPermutationResult",
    "GroupBootstrapResult",
    "HolmEndpoint",
    "NonFiniteInputError",
    "PermutationResult",
    "ResamplingIndices",
    "StatisticsInputError",
    "array_raw_sha256",
    "crossed_group_bootstrap_contrasts",
    "exact_crossed_group_permutation",
    "exact_crossed_spearman_permutation",
    "generate_resampling_indices",
    "group_bootstrap_contrasts",
    "holm_adjust",
    "index_array_sha256",
    "load_resampling_indices_npz",
    "save_resampling_indices_npz",
    "spearman_crossed_ci",
    "two_sided_permutation_test",
]
