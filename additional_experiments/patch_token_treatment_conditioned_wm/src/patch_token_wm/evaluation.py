"""Fold-safe representation summaries, probes, uncertainty, and decisions.

This module is deliberately independent of the world-model implementation.  Its
public fitting APIs make the information boundary visible:

* the attention-free token summarizer fits PCA from outer-train tokens only and
  has no label argument;
* Ridge/logistic candidate models fit on outer train, are selected on outer
  validation, are never refit, and expose a single-use outer-test prediction;
* bootstrap uncertainty resamples paired patients within outer folds and then
  evaluates the pooled patients (folds are not statistical replicates).

The historical response-state audit defines its descriptive calibration slope
as ``Cov(y, y_hat) / Var(y)``.  ``regression_metrics`` keeps that orientation so
that results remain directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler


VISITS = 4
TOKENS_PER_VISIT = 500
TOKEN_DIM = 128
LOCKED_PCA_COMPONENTS = 64
PRIMARY_SUMMARY_DIM = TOKEN_DIM + LOCKED_PCA_COMPONENTS

RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SELECTION_ATOL = 1e-12
MIN_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_SEEDS = (2026, 3026)

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
INCOMPLETE_FINAL = "INCOMPLETE_NO_SCIENTIFIC_CLASSIFICATION"

FINAL_A = "PATCH_WORLD_MODEL_BREAKTHROUGH"
FINAL_B = "PATCH_DYNAMICS_BUT_NO_COMPLEMENTARITY"
FINAL_C = "RESPONSE_ONLY_GAIN"
FINAL_D = "POOLED_LOCAL_REMAINS_SUFFICIENT"


def _numeric_matrix(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_features: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric matrix") from error
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must have non-empty shape [N,F]; got {matrix.shape}")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(f"{name} has {matrix.shape[0]} rows; expected {expected_rows}")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {matrix.shape[1]} features; expected {expected_features}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _numeric_vector(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    minimum_rows: int = 1,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if vector.ndim != 1 or len(vector) < minimum_rows:
        raise ValueError(
            f"{name} must be one-dimensional with at least {minimum_rows} rows"
        )
    if expected_rows is not None and len(vector) != expected_rows:
        raise ValueError(f"{name} has {len(vector)} rows; expected {expected_rows}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def _binary_labels(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    require_both_classes: bool = True,
) -> np.ndarray:
    numeric = _numeric_vector(
        values, name=name, expected_rows=expected_rows, minimum_rows=2
    )
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary 0/1 labels")
    labels = numeric.astype(np.int64)
    if require_both_classes and set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _probabilities(
    values: Any, *, name: str, expected_rows: int | None = None
) -> np.ndarray:
    probabilities = _numeric_vector(
        values, name=name, expected_rows=expected_rows, minimum_rows=2
    )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return probabilities


def _locked_grid(
    values: Iterable[float], *, locked: tuple[float, ...], name: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers")
    try:
        supplied = tuple(sorted({float(value) for value in values}))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an iterable of numbers") from error
    if supplied != locked:
        raise ValueError(f"{name} is locked to {locked}; got {supplied}")
    return supplied


def _token_tensor(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numeric")
    expected_tail = (VISITS, TOKENS_PER_VISIT, TOKEN_DIM)
    if raw.ndim != 4 or tuple(raw.shape[1:]) != expected_tail or raw.shape[0] == 0:
        raise ValueError(f"{name} must have shape [N,4,500,128]; got {raw.shape}")
    # float32 bounds the memory footprint of the 64,000-D PCA matrix.  The exact
    # weighted mean below explicitly accumulates in float64.
    tokens = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(tokens).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return tokens


def _broadcast_fractional_weights(
    values: Any, *, n_patients: int, name: str
) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    target = (n_patients, VISITS, TOKENS_PER_VISIT)
    accepted = {
        (TOKENS_PER_VISIT,),
        (VISITS, TOKENS_PER_VISIT),
        (1, 1, TOKENS_PER_VISIT),
        (1, VISITS, TOKENS_PER_VISIT),
        target,
    }
    if raw.shape not in accepted:
        raise ValueError(
            f"{name} must be broadcastable from [500], [4,500], or "
            f"[N,4,500]; got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError(f"{name} must contain fractional weights in [0,1]")
    weights = np.broadcast_to(raw, target)
    totals = weights.sum(axis=-1, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise ValueError(f"{name} must have positive total weight for every visit")
    return weights


def _normalized_fractional_weights(
    values: Any, *, n_patients: int, name: str
) -> np.ndarray:
    weights = _broadcast_fractional_weights(values, n_patients=n_patients, name=name)
    return weights / weights.sum(axis=-1, keepdims=True, dtype=np.float64)


def fractional_weighted_token_mean(tokens: Any, fractional_weights: Any) -> np.ndarray:
    """Return the exact fractional-overlap weighted token mean ``[N,4,128]``.

    Weights may be shared across all patients/visits (``[500]``), shared across
    patients (``[4,500]``), or supplied per patient/visit (``[N,4,500]``).
    Accumulation is float64 and is exactly ``sum(w*x) / sum(w)``; there is no
    thresholding, token attention, label use, or unweighted fallback.
    """

    token_array = _token_tensor(tokens, name="tokens")
    normalized = _normalized_fractional_weights(
        fractional_weights,
        n_patients=len(token_array),
        name="fractional_weights",
    )
    means = np.einsum(
        "ntk,ntkd->ntd",
        normalized,
        token_array.astype(np.float64, copy=False),
        optimize=True,
    )
    if means.shape != (len(token_array), VISITS, TOKEN_DIM):
        raise AssertionError("weighted token mean shape drifted")
    return means


def _weighted_flattened_tokens(
    tokens: np.ndarray, fractional_weights: Any
) -> np.ndarray:
    normalized = _normalized_fractional_weights(
        fractional_weights,
        n_patients=len(tokens),
        name="fractional_weights",
    )
    # sqrt(normalized weight) makes ordinary Euclidean PCA on the flattened
    # vector equal to fractional-weighted Euclidean geometry over token cells.
    # The exact weighted mean intentionally continues to use normalized w.
    spatial_scale = np.sqrt(normalized).astype(np.float32, copy=False)
    weighted = tokens * spatial_scale[..., None]
    matrix = weighted.reshape(len(tokens) * VISITS, TOKENS_PER_VISIT * TOKEN_DIM)
    if not np.isfinite(matrix).all():
        raise ValueError("weighted flattened tokens contain NaN or infinity")
    return matrix


@dataclass(frozen=True)
class TokenSummaryParts:
    """The three attention-free representation views for a token tensor."""

    weighted_mean: np.ndarray
    pca_scores: np.ndarray
    primary: np.ndarray


class FoldSafeTokenSummarizer:
    """Fit a locked 64-D spatial PCA on one outer fold's training patients.

    PCA rows are patient-visits and PCA columns are the flattened spatial field
    ``tokens * sqrt(fractional_weight / total_fractional_weight)``.  The primary
    output concatenates the independently computed 128-D weighted mean with the
    64-D PCA score.  ``fit`` deliberately contains no target/label parameter.
    """

    def __init__(
        self,
        *,
        outer_fold: int | str,
        n_components: int = LOCKED_PCA_COMPONENTS,
        random_state: int = 0,
    ) -> None:
        if isinstance(n_components, (bool, np.bool_)) or int(n_components) != int(
            LOCKED_PCA_COMPONENTS
        ):
            raise ValueError(
                f"token PCA is locked to {LOCKED_PCA_COMPONENTS} components"
            )
        if isinstance(outer_fold, str) and not outer_fold.strip():
            raise ValueError("outer_fold may not be empty")
        self.outer_fold = outer_fold
        self.n_components = LOCKED_PCA_COMPONENTS
        self.random_state = int(random_state)
        self._pca: PCA | None = None
        self._provenance: dict[str, Any] | None = None

    @property
    def fitted(self) -> bool:
        return self._pca is not None

    @property
    def provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            raise RuntimeError("token summarizer has not been fitted")
        # JSON round-trip prevents callers from mutating nested internal state.
        return json.loads(json.dumps(self._provenance, sort_keys=True))

    @property
    def pca_mean_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("token summarizer has not been fitted")
        return np.asarray(self._pca.mean_).copy()

    @property
    def pca_components_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("token summarizer has not been fitted")
        return np.asarray(self._pca.components_).copy()

    def fit(
        self,
        train_tokens: Any,
        train_fractional_weights: Any,
        *,
        train_patient_ids: Sequence[str] | None = None,
        split: str = "train",
    ) -> "FoldSafeTokenSummarizer":
        """Fit once from outer-train tokens only.

        The explicit ``split='train'`` assertion is provenance hardening, not a
        mechanism for relabeling validation/test data.  The caller remains
        responsible for obtaining these arrays from the locked fold manifest.
        """

        if split != "train":
            raise ValueError("PCA fitting is permitted only for split='train'")
        if self._pca is not None:
            raise RuntimeError(
                "token summarizer is single-fit; create a new fold instance"
            )
        tokens = _token_tensor(train_tokens, name="train_tokens")
        patient_hash: str | None = None
        if train_patient_ids is not None:
            ids = tuple(str(value) for value in train_patient_ids)
            if len(ids) != len(tokens):
                raise ValueError(
                    "train_patient_ids row count differs from train_tokens"
                )
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("train_patient_ids must be non-empty and unique")
            patient_hash = hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()

        matrix = _weighted_flattened_tokens(tokens, train_fractional_weights)
        if matrix.shape[0] <= LOCKED_PCA_COMPONENTS:
            raise ValueError(
                "outer-train must provide more than 64 patient-visit states so "
                "the centered PCA can identify 64 components"
            )
        pca = PCA(
            n_components=LOCKED_PCA_COMPONENTS,
            svd_solver="randomized",
            random_state=self.random_state,
            iterated_power=4,
            whiten=False,
        )
        pca.fit(matrix)
        if pca.components_.shape != (
            LOCKED_PCA_COMPONENTS,
            TOKENS_PER_VISIT * TOKEN_DIM,
        ):
            raise AssertionError("locked PCA component shape drifted")
        if not all(
            np.isfinite(value).all()
            for value in (pca.mean_, pca.components_, pca.explained_variance_ratio_)
        ):
            raise FloatingPointError("token PCA fit produced non-finite state")

        component_hash = hashlib.sha256(
            np.ascontiguousarray(pca.components_).view(np.uint8)
        ).hexdigest()
        self._pca = pca
        self._provenance = {
            "schema_version": 1,
            "outer_fold": self.outer_fold,
            "fit_scope": "outer_train_only",
            "fit_split_assertion": split,
            "labels_used": False,
            "attention_used": False,
            "token_shape": [VISITS, TOKENS_PER_VISIT, TOKEN_DIM],
            "n_train_patients": int(len(tokens)),
            "n_train_patient_visits": int(matrix.shape[0]),
            "pca_input_dim": int(matrix.shape[1]),
            "pca_components": LOCKED_PCA_COMPONENTS,
            "pca_solver": "randomized",
            "pca_random_state": self.random_state,
            "pca_input": "flatten(tokens * sqrt(normalized_fractional_weights))",
            "weighted_mean": "sum(weight * token) / sum(weight)",
            "primary_summary": "weighted_mean_128_then_train_fold_pca_64",
            "primary_summary_dim": PRIMARY_SUMMARY_DIM,
            "train_patient_order_sha256": patient_hash,
            "pca_components_sha256": component_hash,
        }
        return self

    def transform_parts(
        self, tokens: Any, fractional_weights: Any
    ) -> TokenSummaryParts:
        if self._pca is None:
            raise RuntimeError("token summarizer must be fitted before transform")
        token_array = _token_tensor(tokens, name="tokens")
        weighted_mean = fractional_weighted_token_mean(token_array, fractional_weights)
        matrix = _weighted_flattened_tokens(token_array, fractional_weights)
        pca_scores = np.asarray(self._pca.transform(matrix), dtype=np.float64).reshape(
            len(token_array), VISITS, LOCKED_PCA_COMPONENTS
        )
        primary = np.concatenate((weighted_mean, pca_scores), axis=-1)
        if primary.shape != (len(token_array), VISITS, PRIMARY_SUMMARY_DIM):
            raise AssertionError("primary attention-free token summary shape drifted")
        if not np.isfinite(primary).all():
            raise FloatingPointError("token summary contains NaN or infinity")
        return TokenSummaryParts(
            weighted_mean=weighted_mean,
            pca_scores=pca_scores,
            primary=primary,
        )

    def transform(self, tokens: Any, fractional_weights: Any) -> np.ndarray:
        return self.transform_parts(tokens, fractional_weights).primary

    def fit_transform(
        self,
        train_tokens: Any,
        train_fractional_weights: Any,
        *,
        train_patient_ids: Sequence[str] | None = None,
        split: str = "train",
    ) -> np.ndarray:
        self.fit(
            train_tokens,
            train_fractional_weights,
            train_patient_ids=train_patient_ids,
            split=split,
        )
        return self.transform(train_tokens, train_fractional_weights)


def _safe_correlation(
    function: Callable[..., Any], truth: np.ndarray, prediction: np.ndarray
) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0.0 or np.ptp(prediction) == 0.0:
        return math.nan
    value = float(function(truth, prediction).statistic)
    return value if math.isfinite(value) else math.nan


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, int | float]:
    """FTV/DeltaFTV metrics on the natural target scale."""

    truth = _numeric_vector(y_true, name="y_true", minimum_rows=2)
    prediction = _numeric_vector(
        y_pred, name="y_pred", expected_rows=len(truth), minimum_rows=2
    )
    residual = prediction - truth
    target_centered = truth - np.mean(truth)
    target_ss = float(np.sum(np.square(target_centered)))
    target_variance = float(np.var(truth, ddof=0))
    prediction_variance = float(np.var(prediction, ddof=0))
    if target_ss > 0.0:
        natural_r2 = 1.0 - float(np.sum(np.square(residual))) / target_ss
        covariance = float(
            np.mean((truth - np.mean(truth)) * (prediction - np.mean(prediction)))
        )
        calibration_slope = covariance / target_variance
        variance_ratio = prediction_variance / target_variance
    else:
        natural_r2 = calibration_slope = variance_ratio = math.nan
    return {
        "n": int(len(truth)),
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "pearson": _safe_correlation(pearsonr, truth, prediction),
        "natural_r2": natural_r2,
        "rmse": float(math.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "prediction_target_variance_ratio": variance_ratio,
        "calibration_slope": calibration_slope,
    }


def pcr_metrics(y_true: Any, probabilities: Any) -> dict[str, int | float]:
    """Return the locked MRI-only/complementarity pCR probability metrics."""

    labels = _binary_labels(y_true, name="pCR labels")
    probability = _probabilities(
        probabilities, name="pCR probabilities", expected_rows=len(labels)
    )
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probability)),
        # This is sklearn average precision (the project's established AUPRC
        # definition), not trapezoidal PR-curve integration.
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(np.mean(np.square(probability - labels))),
    }


@dataclass(frozen=True)
class RidgeCandidateScore:
    alpha: float
    validation_mse: float


@dataclass
class FoldSafeRidgeProbe:
    scaler: StandardScaler
    model: Ridge
    selected_alpha: float
    validation_mse: float
    candidate_scores: tuple[RidgeCandidateScore, ...]
    feature_dim: int
    train_rows: int
    validation_rows: int
    outer_fold: int | str | None = None
    _test_prediction_calls: int = field(default=0, init=False, repr=False)

    @property
    def test_prediction_calls(self) -> int:
        return self._test_prediction_calls

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "fit_scope": "outer_train_only",
            "selection_scope": "outer_validation_mse_only",
            "refit_after_selection": False,
            "test_prediction_calls": self._test_prediction_calls,
            "selected_alpha": self.selected_alpha,
            "alpha_grid": list(RIDGE_ALPHAS),
            "tie_break": "smallest_alpha_within_1e-12",
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_dim": self.feature_dim,
        }

    def predict_test_once(self, test_features: Any) -> np.ndarray:
        matrix = _numeric_matrix(
            test_features,
            name="outer-test Ridge features",
            expected_features=self.feature_dim,
        )
        if self._test_prediction_calls:
            raise RuntimeError("outer-test Ridge prediction is single-use")
        self._test_prediction_calls += 1
        prediction = np.asarray(
            self.model.predict(self.scaler.transform(matrix)), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("outer-test Ridge prediction is non-finite")
        return prediction


def fit_fold_safe_ridge(
    train_features: Any,
    train_targets: Any,
    validation_features: Any,
    validation_targets: Any,
    *,
    alphas: Iterable[float] = RIDGE_ALPHAS,
    outer_fold: int | str | None = None,
) -> FoldSafeRidgeProbe:
    """Fit on train and select on validation; test is absent by signature."""

    train_x = _numeric_matrix(train_features, name="outer-train Ridge features")
    validation_x = _numeric_matrix(
        validation_features,
        name="outer-validation Ridge features",
        expected_features=train_x.shape[1],
    )
    train_y = _numeric_vector(
        train_targets,
        name="outer-train Ridge target",
        expected_rows=len(train_x),
        minimum_rows=2,
    )
    validation_y = _numeric_vector(
        validation_targets,
        name="outer-validation Ridge target",
        expected_rows=len(validation_x),
        minimum_rows=2,
    )
    grid = _locked_grid(alphas, locked=RIDGE_ALPHAS, name="Ridge alpha grid")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidates: list[tuple[RidgeCandidateScore, Ridge]] = []
    for alpha in grid:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        ).fit(train_scaled, train_y)
        score = float(
            mean_squared_error(validation_y, model.predict(validation_scaled))
        )
        if not math.isfinite(score):
            raise FloatingPointError(
                f"validation Ridge MSE is non-finite for alpha={alpha}"
            )
        candidates.append((RidgeCandidateScore(alpha, score), model))
    best = min(item[0].validation_mse for item in candidates)
    eligible = [
        item for item in candidates if item[0].validation_mse <= best + SELECTION_ATOL
    ]
    selected, model = min(eligible, key=lambda item: item[0].alpha)
    return FoldSafeRidgeProbe(
        scaler=scaler,
        model=model,
        selected_alpha=selected.alpha,
        validation_mse=selected.validation_mse,
        candidate_scores=tuple(item[0] for item in candidates),
        feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
        outer_fold=outer_fold,
    )


@dataclass(frozen=True)
class LogisticCandidateScore:
    c_value: float
    validation_auroc: float


@dataclass
class FoldSafeLogisticProbe:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_auroc: float
    candidate_scores: tuple[LogisticCandidateScore, ...]
    feature_dim: int
    train_rows: int
    validation_rows: int
    outer_fold: int | str | None = None
    _test_prediction_calls: int = field(default=0, init=False, repr=False)

    @property
    def test_prediction_calls(self) -> int:
        return self._test_prediction_calls

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "fit_scope": "outer_train_only",
            "selection_scope": "outer_validation_auroc_only",
            "refit_after_selection": False,
            "test_prediction_calls": self._test_prediction_calls,
            "selected_c": self.selected_c,
            "c_grid": list(LOGISTIC_CS),
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": None,
            "tie_break": "smallest_C_within_1e-12",
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_dim": self.feature_dim,
        }

    def predict_test_once(self, test_features: Any) -> np.ndarray:
        matrix = _numeric_matrix(
            test_features,
            name="outer-test logistic features",
            expected_features=self.feature_dim,
        )
        if self._test_prediction_calls:
            raise RuntimeError("outer-test logistic prediction is single-use")
        self._test_prediction_calls += 1
        if tuple(self.model.classes_.tolist()) != (0, 1):
            raise RuntimeError("logistic class order drifted")
        probability = np.asarray(
            self.model.predict_proba(self.scaler.transform(matrix))[:, 1],
            dtype=np.float64,
        )
        return _probabilities(
            probability,
            name="outer-test logistic probabilities",
            expected_rows=len(matrix),
        )


def fit_fold_safe_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    *,
    c_grid: Iterable[float] = LOGISTIC_CS,
    outer_fold: int | str | None = None,
    random_state: int = 0,
) -> FoldSafeLogisticProbe:
    """Fit locked L2/liblinear candidates and select validation AUROC only."""

    train_x = _numeric_matrix(train_features, name="outer-train logistic features")
    validation_x = _numeric_matrix(
        validation_features,
        name="outer-validation logistic features",
        expected_features=train_x.shape[1],
    )
    train_y = _binary_labels(
        train_labels, name="outer-train pCR labels", expected_rows=len(train_x)
    )
    validation_y = _binary_labels(
        validation_labels,
        name="outer-validation pCR labels",
        expected_rows=len(validation_x),
    )
    grid = _locked_grid(c_grid, locked=LOGISTIC_CS, name="logistic C grid")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidates: list[tuple[LogisticCandidateScore, LogisticRegression]] = []
    for c_value in grid:
        model = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver="liblinear",
            class_weight=None,
            max_iter=10_000,
            tol=1e-8,
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_scaled, train_y)
        probability = model.predict_proba(validation_scaled)[:, 1]
        score = float(roc_auc_score(validation_y, probability))
        if not math.isfinite(score):
            raise FloatingPointError(
                f"validation logistic AUROC is non-finite for C={c_value}"
            )
        candidates.append((LogisticCandidateScore(c_value, score), model))
    best = max(item[0].validation_auroc for item in candidates)
    eligible = [
        item for item in candidates if item[0].validation_auroc >= best - SELECTION_ATOL
    ]
    selected, model = min(eligible, key=lambda item: item[0].c_value)
    return FoldSafeLogisticProbe(
        scaler=scaler,
        model=model,
        selected_c=selected.c_value,
        validation_auroc=selected.validation_auroc,
        candidate_scores=tuple(item[0] for item in candidates),
        feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
        outer_fold=outer_fold,
    )


@dataclass(frozen=True)
class OuterFoldPrediction:
    predictions: np.ndarray
    selected_hyperparameter: float
    validation_score: float
    provenance: Mapping[str, Any]


def run_outer_fold_ridge(
    train_features: Any,
    train_targets: Any,
    validation_features: Any,
    validation_targets: Any,
    test_features: Any,
    *,
    outer_fold: int | str,
) -> OuterFoldPrediction:
    """Run one train/validation/test Ridge fold, predicting test exactly once."""

    fitted = fit_fold_safe_ridge(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        outer_fold=outer_fold,
    )
    predictions = fitted.predict_test_once(test_features)
    if fitted.test_prediction_calls != 1:
        raise AssertionError("outer-test Ridge prediction count drifted")
    return OuterFoldPrediction(
        predictions=predictions,
        selected_hyperparameter=fitted.selected_alpha,
        validation_score=fitted.validation_mse,
        provenance=fitted.provenance,
    )


def run_outer_fold_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    test_features: Any,
    *,
    outer_fold: int | str,
    random_state: int = 0,
) -> OuterFoldPrediction:
    """Run one train/validation/test pCR fold, predicting test exactly once."""

    fitted = fit_fold_safe_logistic(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        outer_fold=outer_fold,
        random_state=random_state,
    )
    predictions = fitted.predict_test_once(test_features)
    if fitted.test_prediction_calls != 1:
        raise AssertionError("outer-test logistic prediction count drifted")
    return OuterFoldPrediction(
        predictions=predictions,
        selected_hyperparameter=fitted.selected_c,
        validation_score=fitted.validation_auroc,
        provenance=fitted.provenance,
    )


MetricFunction = Callable[[np.ndarray, np.ndarray], float]
MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class PairedBootstrapResult:
    summary: pd.DataFrame
    draws: pd.DataFrame
    n_patients: int
    fold_sizes: Mapping[str, int]


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    patient_col: str,
    fold_col: str,
    truth_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    required = {patient_col, fold_col, truth_col, prediction_col}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{name} is missing columns: {missing}")
    output = frame.loc[:, [patient_col, fold_col, truth_col, prediction_col]].copy()
    if output.isna().any().any():
        raise ValueError(f"{name} contains missing pairing or prediction values")
    output[patient_col] = output[patient_col].astype(str)
    if output[patient_col].eq("").any() or output[patient_col].duplicated().any():
        raise ValueError(f"{name} must contain exactly one row per non-empty patient")
    output[truth_col] = _numeric_vector(
        output[truth_col].to_numpy(),
        name=f"{name} truth",
        expected_rows=len(output),
        minimum_rows=2,
    )
    output[prediction_col] = _numeric_vector(
        output[prediction_col].to_numpy(),
        name=f"{name} prediction",
        expected_rows=len(output),
        minimum_rows=2,
    )
    return output


def paired_metric_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    metric_functions: Mapping[str, MetricFunction],
    metric_directions: Mapping[str, MetricDirection],
    patient_col: str = "patient_id",
    fold_col: str = "fold",
    truth_col: str = "y_true",
    prediction_col: str = "y_pred",
    n_bootstrap: int = MIN_BOOTSTRAP_REPLICATES,
    confidence_level: float = 0.95,
    seed: int = 260_812,
) -> PairedBootstrapResult:
    """Paired patient bootstrap, sampling within fold and scoring pooled rows.

    A positive ``improvement`` always favors ``comparison``: comparison minus
    reference for higher-is-better metrics, and reference minus comparison for
    lower-is-better metrics.
    """

    if isinstance(n_bootstrap, (bool, np.bool_)) or int(n_bootstrap) != n_bootstrap:
        raise ValueError("n_bootstrap must be an integer")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"formal paired bootstrap requires at least {MIN_BOOTSTRAP_REPLICATES} draws"
        )
    if not math.isfinite(float(confidence_level)) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not metric_functions:
        raise ValueError("metric_functions may not be empty")
    if set(metric_directions) != set(metric_functions):
        raise ValueError("metric_directions must exactly match metric_functions")
    if any(value not in ("higher", "lower") for value in metric_directions.values()):
        raise ValueError("metric directions must be 'higher' or 'lower'")

    reference_frame = _prediction_frame(
        reference,
        name="reference",
        patient_col=patient_col,
        fold_col=fold_col,
        truth_col=truth_col,
        prediction_col=prediction_col,
    ).rename(
        columns={
            fold_col: "reference_fold",
            truth_col: "reference_truth",
            prediction_col: "reference_prediction",
        }
    )
    comparison_frame = _prediction_frame(
        comparison,
        name="comparison",
        patient_col=patient_col,
        fold_col=fold_col,
        truth_col=truth_col,
        prediction_col=prediction_col,
    ).rename(
        columns={
            fold_col: "comparison_fold",
            truth_col: "comparison_truth",
            prediction_col: "comparison_prediction",
        }
    )
    if set(reference_frame[patient_col]) != set(comparison_frame[patient_col]):
        raise ValueError("reference and comparison patient sets must match exactly")
    paired = reference_frame.merge(
        comparison_frame, on=patient_col, how="outer", validate="one_to_one"
    )
    if not np.array_equal(
        paired["reference_fold"].astype(str).to_numpy(),
        paired["comparison_fold"].astype(str).to_numpy(),
    ):
        raise ValueError("reference and comparison fold assignments must match exactly")
    if not np.array_equal(
        paired["reference_truth"].to_numpy(dtype=np.float64),
        paired["comparison_truth"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("reference and comparison truths disagree")
    paired["fold_key"] = paired["reference_fold"].astype(str)
    paired = paired.sort_values(
        ["fold_key", patient_col], kind="mergesort"
    ).reset_index(drop=True)
    truth = paired["reference_truth"].to_numpy(dtype=np.float64)
    reference_prediction = paired["reference_prediction"].to_numpy(dtype=np.float64)
    comparison_prediction = paired["comparison_prediction"].to_numpy(dtype=np.float64)

    def point_metric(function: MetricFunction, prediction: np.ndarray) -> float:
        value = float(function(truth, prediction))
        if not math.isfinite(value):
            raise ValueError("paired point metric is undefined or non-finite")
        return value

    point_values = {
        name: (
            point_metric(function, reference_prediction),
            point_metric(function, comparison_prediction),
        )
        for name, function in metric_functions.items()
    }

    rng = np.random.default_rng(int(seed))
    sampled_blocks: list[np.ndarray] = []
    fold_sizes: dict[str, int] = {}
    folds = paired["fold_key"].to_numpy()
    for fold_name in sorted(np.unique(folds).tolist()):
        positions = np.flatnonzero(folds == fold_name)
        fold_sizes[str(fold_name)] = int(len(positions))
        sampled_blocks.append(
            rng.choice(positions, size=(n_bootstrap, len(positions)), replace=True)
        )
    sampled_indices = np.concatenate(sampled_blocks, axis=1)
    draw_columns: dict[str, np.ndarray] = {
        "bootstrap_index": np.arange(n_bootstrap, dtype=np.int64)
    }
    for metric_name, function in metric_functions.items():
        values = np.full(n_bootstrap, np.nan, dtype=np.float64)
        direction = metric_directions[metric_name]
        for draw_index, indices in enumerate(sampled_indices):
            sampled_truth = truth[indices]
            try:
                reference_value = float(
                    function(sampled_truth, reference_prediction[indices])
                )
                comparison_value = float(
                    function(sampled_truth, comparison_prediction[indices])
                )
            except (ValueError, FloatingPointError):
                continue
            if math.isfinite(reference_value) and math.isfinite(comparison_value):
                values[draw_index] = (
                    comparison_value - reference_value
                    if direction == "higher"
                    else reference_value - comparison_value
                )
        draw_columns[f"{metric_name}_improvement"] = values
    draws = pd.DataFrame(draw_columns)

    alpha = 1.0 - float(confidence_level)
    summary_rows: list[dict[str, Any]] = []
    for metric_name, direction in metric_directions.items():
        reference_value, comparison_value = point_values[metric_name]
        improvement = (
            comparison_value - reference_value
            if direction == "higher"
            else reference_value - comparison_value
        )
        distribution = draws[f"{metric_name}_improvement"].to_numpy(dtype=np.float64)
        finite = distribution[np.isfinite(distribution)]
        summary_rows.append(
            {
                "metric": metric_name,
                "reference": reference_value,
                "comparison": comparison_value,
                "improvement": improvement,
                "ci_lower": (
                    float(np.quantile(finite, alpha / 2.0)) if len(finite) else math.nan
                ),
                "ci_upper": (
                    float(np.quantile(finite, 1.0 - alpha / 2.0))
                    if len(finite)
                    else math.nan
                ),
                "confidence_level": float(confidence_level),
                "n_patients": int(len(paired)),
                "n_folds": int(len(fold_sizes)),
                "n_bootstrap": n_bootstrap,
                "n_valid_bootstrap": int(len(finite)),
                "bootstrap_unit": "patient_within_outer_fold",
                "metric_aggregation": "pooled_patients_not_fold_replicates",
                "ci_method": "percentile",
                "orientation": (
                    "comparison - reference"
                    if direction == "higher"
                    else "reference - comparison (lower is better)"
                ),
                "seed": int(seed),
            }
        )
    return PairedBootstrapResult(
        summary=pd.DataFrame(summary_rows),
        draws=draws,
        n_patients=int(len(paired)),
        fold_sizes=fold_sizes,
    )


def paired_pcr_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    fold_col: str = "fold",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    n_bootstrap: int = MIN_BOOTSTRAP_REPLICATES,
    confidence_level: float = 0.95,
    seed: int = 260_812,
) -> PairedBootstrapResult:
    """Formal paired AUROC/AUPRC/Brier bootstrap."""

    for name, frame in (("reference", reference), ("comparison", comparison)):
        if not isinstance(frame, pd.DataFrame) or probability_col not in frame:
            raise ValueError(
                f"{name} is missing probability column {probability_col!r}"
            )
        _probabilities(
            frame[probability_col].to_numpy(),
            name=f"{name} pCR probabilities",
            expected_rows=len(frame),
        )

    def auroc(labels: np.ndarray, probability: np.ndarray) -> float:
        binary = _binary_labels(labels, name="bootstrap pCR labels")
        return float(roc_auc_score(binary, probability))

    def auprc(labels: np.ndarray, probability: np.ndarray) -> float:
        binary = _binary_labels(labels, name="bootstrap pCR labels")
        return float(average_precision_score(binary, probability))

    def brier(labels: np.ndarray, probability: np.ndarray) -> float:
        binary = _binary_labels(
            labels, name="bootstrap pCR labels", require_both_classes=False
        )
        return float(np.mean(np.square(probability - binary)))

    return paired_metric_bootstrap(
        reference,
        comparison,
        metric_functions={"auroc": auroc, "auprc": auprc, "brier": brier},
        metric_directions={"auroc": "higher", "auprc": "higher", "brier": "lower"},
        patient_col=patient_col,
        fold_col=fold_col,
        truth_col=label_col,
        prediction_col=probability_col,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


@dataclass(frozen=True)
class GateDecision:
    gate: str
    status: Literal["PASS", "FAIL", "INCOMPLETE"]
    success_label: str | None
    reason: str
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "success_label": self.success_label,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class FinalClassification:
    status: Literal["COMPLETE", "INCOMPLETE"]
    label: str
    reason: str
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": self.label,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


def _seed_row(rows: Mapping[Any, Any], seed: int) -> Mapping[str, Any] | None:
    value = rows.get(seed, rows.get(str(seed)))
    return value if isinstance(value, Mapping) else None


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _incomplete(gate: str, reason: str, evidence: Mapping[str, Any]) -> GateDecision:
    return GateDecision(gate, INCOMPLETE, None, reason, evidence)


def evaluate_gate_a(
    dynamics_by_seed: Mapping[Any, Any],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_cells_per_seed: int = 5,
) -> GateDecision:
    """Gate A using the locked non-collapse and shuffled-time thresholds.

    Each seed row is the aggregate emitted by :mod:`patch_token_wm.diagnostics`:
    ``folds`` (or ``finite_cell_count``), ``target_std``, ``prediction_std``, and
    either ``cosine_gain`` (or actual/shuffled cosine) or
    ``normalized_mse_relative_improvement``.  Requiring five finite cells for
    each of the two seeds proves that all ten A1 cells entered the gate.
    """

    if (
        isinstance(expected_cells_per_seed, (bool, np.bool_))
        or int(expected_cells_per_seed) != expected_cells_per_seed
        or int(expected_cells_per_seed) <= 0
    ):
        raise ValueError("expected_cells_per_seed must be a positive integer")
    expected_cells_per_seed = int(expected_cells_per_seed)
    evidence: dict[str, Any] = {
        "seeds": {},
        "expected_total_cells": expected_cells_per_seed * len(expected_seeds),
        "token_std_min": 0.05,
        "cosine_gain_min": 0.05,
        "normalized_mse_relative_improvement_min": 0.05,
    }
    seed_passes: list[bool] = []
    for seed in expected_seeds:
        row = _seed_row(dynamics_by_seed, int(seed))
        if row is None:
            return _incomplete(
                "A",
                f"missing complete token-dynamics evidence for seed {seed}",
                evidence,
            )
        cell_count = row.get("finite_cell_count", row.get("folds"))
        target_std = row.get("target_std", row.get("target_token_std"))
        prediction_std = row.get("prediction_std", row.get("prediction_token_std"))
        cosine_gain = row.get("cosine_gain")
        if not _finite_number(cosine_gain):
            actual = row.get("actual_cosine", row.get("actual_time_cosine"))
            shuffled = row.get("shuffled_cosine", row.get("shuffled_time_cosine"))
            if _finite_number(actual) and _finite_number(shuffled):
                cosine_gain = float(actual) - float(shuffled)
        mse_gain = row.get("normalized_mse_relative_improvement")
        if (
            not _finite_number(cell_count)
            or int(float(cell_count)) != float(cell_count)
            or not _finite_number(target_std)
            or not _finite_number(prediction_std)
            or (not _finite_number(cosine_gain) and not _finite_number(mse_gain))
        ):
            return _incomplete(
                "A",
                f"missing or non-finite locked dynamics fields for seed {seed}",
                evidence,
            )
        finite_cells = int(float(cell_count))
        cosine_value = float(cosine_gain) if _finite_number(cosine_gain) else None
        mse_value = float(mse_gain) if _finite_number(mse_gain) else None
        variance_preserved = float(target_std) >= 0.05 and float(prediction_std) >= 0.05
        exceeds_shuffle = bool(
            (cosine_value is not None and cosine_value >= 0.05)
            or (mse_value is not None and mse_value >= 0.05)
        )
        all_cells_finite = finite_cells == expected_cells_per_seed
        normalized = {
            "finite_cell_count": finite_cells,
            "expected_cell_count": expected_cells_per_seed,
            "all_cells_finite": all_cells_finite,
            "target_std": float(target_std),
            "prediction_std": float(prediction_std),
            "token_variance_preserved": variance_preserved,
            "cosine_gain": cosine_value,
            "normalized_mse_relative_improvement": mse_value,
            "materially_exceeds_shuffled": exceeds_shuffle,
        }
        evidence["seeds"][str(seed)] = normalized
        seed_passes.append(all_cells_finite and variance_preserved and exceeds_shuffle)
    passed = all(seed_passes)
    return GateDecision(
        "A",
        GATE_PASS if passed else GATE_FAIL,
        "PATCH_DYNAMICS_VALID" if passed else None,
        (
            "both seeds have finite, non-collapsed, variance-preserving dynamics above shuffle"
            if passed
            else "at least one seed fails a locked token-dynamics criterion"
        ),
        evidence,
    )


def evaluate_gate_b(
    effects_by_seed: Mapping[Any, Any], *, expected_seeds: Sequence[int] = DEFAULT_SEEDS
) -> GateDecision:
    """Gate B response preservation.

    Static and DeltaFTV A1-A0 Spearman must each be strictly greater than
    -0.03 in every seed.  This is the plan's locked operational definition of
    "no systematic DeltaFTV degradation."
    """

    static: list[float] = []
    delta: list[float] = []
    evidence: dict[str, Any] = {
        "static_threshold_strict": -0.03,
        "delta_threshold_strict": -0.03,
        "seeds": {},
    }
    for seed in expected_seeds:
        row = _seed_row(effects_by_seed, int(seed))
        fields = ("static_ftv_spearman_delta", "delta_ftv_spearman_delta")
        if row is None or any(field not in row for field in fields):
            return _incomplete(
                "B", f"missing response effects for seed {seed}", evidence
            )
        if not all(_finite_number(row[field]) for field in fields):
            return _incomplete(
                "B", f"non-finite response effects for seed {seed}", evidence
            )
        static_value = float(row[fields[0]])
        delta_value = float(row[fields[1]])
        static.append(static_value)
        delta.append(delta_value)
        evidence["seeds"][str(seed)] = {
            fields[0]: static_value,
            fields[1]: delta_value,
        }
    static_preserved = all(value > -0.03 for value in static)
    delta_preserved = all(value > -0.03 for value in delta)
    evidence["static_preserved_each_seed"] = static_preserved
    evidence["delta_no_systematic_degradation"] = delta_preserved
    passed = static_preserved and delta_preserved
    return GateDecision(
        "B",
        GATE_PASS if passed else GATE_FAIL,
        None,
        (
            "response preservation criteria pass"
            if passed
            else "response preservation criteria fail"
        ),
        evidence,
    )


def evaluate_gate_c(
    effects_by_seed: Mapping[Any, Any], *, expected_seeds: Sequence[int] = DEFAULT_SEEDS
) -> GateDecision:
    """Gate C: one endpoint positive in both seeds with mean effect >= +0.03."""

    endpoints = (
        "static_ftv_spearman_delta",
        "delta_ftv_spearman_delta",
        "mri_pcr_auroc_delta",
    )
    rows: dict[int, Mapping[str, Any]] = {}
    for seed in expected_seeds:
        row = _seed_row(effects_by_seed, int(seed))
        if row is None or any(endpoint not in row for endpoint in endpoints):
            return _incomplete(
                "C",
                f"missing spatial-gain endpoints for seed {seed}",
                {"endpoints": {}},
            )
        if not all(_finite_number(row[endpoint]) for endpoint in endpoints):
            return _incomplete(
                "C",
                f"non-finite spatial-gain endpoints for seed {seed}",
                {"endpoints": {}},
            )
        rows[int(seed)] = row
    endpoint_evidence: dict[str, Any] = {}
    for endpoint in endpoints:
        values = [float(rows[int(seed)][endpoint]) for seed in expected_seeds]
        positive_both = all(value > 0.0 for value in values)
        mean_effect = float(np.mean(values))
        endpoint_evidence[endpoint] = {
            "by_seed": {
                str(seed): value for seed, value in zip(expected_seeds, values)
            },
            "positive_both": positive_both,
            "mean_effect": mean_effect,
            "passes": positive_both and mean_effect >= 0.03,
        }
    passed = any(row["passes"] for row in endpoint_evidence.values())
    return GateDecision(
        "C",
        GATE_PASS if passed else GATE_FAIL,
        "PATCH_STATE_ADDS_INFORMATION" if passed else None,
        (
            "at least one endpoint improves in both seeds with mean effect >= +0.03"
            if passed
            else "no endpoint meets the two-seed +0.03 mean-effect rule"
        ),
        {"threshold": 0.03, "endpoints": endpoint_evidence},
    )


def _timing_record(
    timings: Mapping[str, Any], canonical: str
) -> Mapping[str, Any] | None:
    aliases = {
        "T0-T1": ("T0-T1", "T0_to_T1", "T0–T1"),
        "T0-T2": ("T0-T2", "T0_to_T2", "T0–T2"),
    }
    for key in aliases[canonical]:
        value = timings.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def evaluate_gate_d(
    complementarity_by_timing: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> GateDecision:
    """Gate D: at T0-T1 or T0-T2, both seed AUROC increments are positive.

    A paired pooled-patient CI lower bound is recorded when supplied but is not
    made a hard requirement, exactly matching the prompt's "preferably" clause.
    A timing record may contain seed keys directly or under ``seed_effects``.
    """

    evidence: dict[str, Any] = {"timings": {}}
    incomplete_timings: list[str] = []
    for timing in ("T0-T1", "T0-T2"):
        record = _timing_record(complementarity_by_timing, timing)
        if record is None:
            incomplete_timings.append(timing)
            continue
        seed_effects = record.get("seed_effects", record)
        if not isinstance(seed_effects, Mapping):
            incomplete_timings.append(timing)
            continue
        values: list[float] = []
        complete = True
        for seed in expected_seeds:
            value = seed_effects.get(seed, seed_effects.get(str(seed)))
            if not _finite_number(value):
                complete = False
                break
            values.append(float(value))
        if not complete:
            incomplete_timings.append(timing)
            continue
        ci_lower = record.get("bootstrap_ci_lower")
        ci_support: bool | None = None
        if _finite_number(ci_lower):
            ci_support = float(ci_lower) > 0.0
        passed = all(value > 0.0 for value in values)
        evidence["timings"][timing] = {
            "by_seed": {
                str(seed): value for seed, value in zip(expected_seeds, values)
            },
            "both_seed_point_estimates_positive": passed,
            "bootstrap_ci_lower": float(ci_lower) if _finite_number(ci_lower) else None,
            "bootstrap_ci_excludes_zero": ci_support,
            "ci_is_preferred_not_hard": True,
        }
        if passed:
            return GateDecision(
                "D",
                GATE_PASS,
                "PATCH_STATE_COMPLEMENTARITY_SUPPORTED",
                f"both seed point estimates are positive at {timing}",
                evidence,
            )
    if incomplete_timings:
        evidence["incomplete_timings"] = incomplete_timings
        return _incomplete(
            "D",
            "no complete passing timing and at least one eligible timing is incomplete",
            evidence,
        )
    return GateDecision(
        "D",
        GATE_FAIL,
        None,
        "neither T0-T1 nor T0-T2 has positive increments in both seeds",
        evidence,
    )


def _get_gate(
    decisions: Mapping[str, GateDecision | Mapping[str, Any]], letter: str
) -> GateDecision | None:
    candidate = decisions.get(letter, decisions.get(f"gate_{letter.lower()}"))
    if isinstance(candidate, GateDecision):
        return candidate
    if isinstance(candidate, Mapping):
        status = candidate.get("status")
        if status in (GATE_PASS, GATE_FAIL, INCOMPLETE):
            return GateDecision(
                gate=letter,
                status=status,
                success_label=candidate.get("success_label"),
                reason=str(candidate.get("reason", "")),
                evidence=(
                    candidate.get("evidence", {})
                    if isinstance(candidate.get("evidence", {}), Mapping)
                    else {}
                ),
            )
    return None


def classify_final(
    decisions: Mapping[str, GateDecision | Mapping[str, Any]],
) -> FinalClassification:
    """Apply the preregistered A/B/C/D hierarchy without filling missing cells.

    Missing formal evidence is always incomplete.  Once all formal gate cells
    are present, the locked precedence is breakthrough, response-only gain,
    pCR gain without complementarity, then pooled LOCAL remains sufficient.
    """

    gates = {letter: _get_gate(decisions, letter) for letter in "ABCD"}
    if any(value is None or value.status == INCOMPLETE for value in gates.values()):
        return FinalClassification(
            status=INCOMPLETE,
            label=INCOMPLETE_FINAL,
            reason="all four gate decisions must be complete before final classification",
            evidence={
                letter: None if value is None else value.status
                for letter, value in gates.items()
            },
        )
    statuses = {letter: gates[letter].status for letter in "ABCD"}  # type: ignore[union-attr]
    if all(statuses[letter] == GATE_PASS for letter in "ABCD"):
        return FinalClassification(
            "COMPLETE", FINAL_A, "Gates A, B, C, and D pass", statuses
        )

    gate_c = gates["C"]
    endpoint_rows = gate_c.evidence.get("endpoints", {})  # type: ignore[union-attr]
    response_positive = False
    pcr_meaningful = False
    if isinstance(endpoint_rows, Mapping):
        response_positive = any(
            isinstance(endpoint_rows.get(name), Mapping)
            and endpoint_rows[name].get("positive_both") is True
            for name in ("static_ftv_spearman_delta", "delta_ftv_spearman_delta")
        )
        pcr_row = endpoint_rows.get("mri_pcr_auroc_delta")
        pcr_meaningful = isinstance(pcr_row, Mapping) and pcr_row.get("passes") is True
    augmented = {
        **statuses,
        "response_positive_both_seeds": response_positive,
        "mri_pcr_meaningful_two_seed_gain": pcr_meaningful,
    }
    if (
        statuses["A"] == GATE_PASS
        and statuses["B"] == GATE_PASS
        and response_positive
        and not pcr_meaningful
    ):
        return FinalClassification(
            "COMPLETE",
            FINAL_C,
            "A/B pass with a two-seed response gain but no meaningful MRI-only pCR gain",
            augmented,
        )
    if (
        statuses["A"] == GATE_PASS
        and statuses["B"] == GATE_PASS
        and statuses["C"] == GATE_PASS
        and statuses["D"] == GATE_FAIL
        and pcr_meaningful
    ):
        return FinalClassification(
            "COMPLETE",
            FINAL_B,
            "A/B/C pass through MRI-only pCR gain but Gate D fails",
            augmented,
        )
    return FinalClassification(
        status="COMPLETE",
        label=FINAL_D,
        reason="complete formal evidence does not meet the higher-precedence gain classes",
        evidence=augmented,
    )


def evaluate_all_gates(
    *,
    dynamics_by_seed: Mapping[Any, Any],
    effects_by_seed: Mapping[Any, Any],
    complementarity_by_timing: Mapping[str, Any],
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Evaluate all gates and the final label in one JSON-friendly payload."""

    gates = {
        "A": evaluate_gate_a(dynamics_by_seed, expected_seeds=expected_seeds),
        "B": evaluate_gate_b(effects_by_seed, expected_seeds=expected_seeds),
        "C": evaluate_gate_c(effects_by_seed, expected_seeds=expected_seeds),
        "D": evaluate_gate_d(complementarity_by_timing, expected_seeds=expected_seeds),
    }
    final = classify_final(gates)
    return {
        "gates": {letter: decision.as_dict() for letter, decision in gates.items()},
        "final": final.as_dict(),
    }


__all__ = [
    "DEFAULT_SEEDS",
    "FINAL_A",
    "FINAL_B",
    "FINAL_C",
    "FINAL_D",
    "FoldSafeLogisticProbe",
    "FoldSafeRidgeProbe",
    "FoldSafeTokenSummarizer",
    "GATE_FAIL",
    "GATE_PASS",
    "GateDecision",
    "INCOMPLETE",
    "INCOMPLETE_FINAL",
    "LOCKED_PCA_COMPONENTS",
    "LOGISTIC_CS",
    "LogisticCandidateScore",
    "MIN_BOOTSTRAP_REPLICATES",
    "OuterFoldPrediction",
    "PRIMARY_SUMMARY_DIM",
    "PairedBootstrapResult",
    "RIDGE_ALPHAS",
    "RidgeCandidateScore",
    "TOKEN_DIM",
    "TOKENS_PER_VISIT",
    "TokenSummaryParts",
    "VISITS",
    "classify_final",
    "evaluate_all_gates",
    "evaluate_gate_a",
    "evaluate_gate_b",
    "evaluate_gate_c",
    "evaluate_gate_d",
    "fit_fold_safe_logistic",
    "fit_fold_safe_ridge",
    "fractional_weighted_token_mean",
    "paired_metric_bootstrap",
    "paired_pcr_bootstrap",
    "pcr_metrics",
    "regression_metrics",
    "run_outer_fold_logistic",
    "run_outer_fold_ridge",
]
