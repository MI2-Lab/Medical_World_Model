from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Filesystem and tensor-construction settings.

    The crop is ordered as ``(Z, Y, X)``. Raw NIfTI arrays are read as
    ``(X, Y, Z, phase)`` and converted to model tensors in ``(C, Z, Y, X)``.
    """

    ispy2_root: str = "/data/data/Preprocessed/I-SPY2"
    ispy2_labels: str = "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
    ispy1_root: str | None = "/data/data/Preprocessed/I-SPY1"
    ispy1_labels: str | None = "/data/data/Preprocessed/I-SPY1/clinical_labels_complete4visits.csv"
    tensor_cache: str = "/data/data/Preprocessed/I-SPY2/_corejepa_clean_dce8"
    response_cache: str = "/data/data/Preprocessed/I-SPY2/corejepa_response_features.npz"
    breastdcedl_metadata_csv: str | None = None
    crop_size: tuple[int, int, int] = (32, 96, 96)
    phase_policy: str = "adaptive_early_late"
    response_phase_policy: str = "breastdcedl"
    auto_roi_fallback: bool = True
    min_roi_capture: float = 0.5
    legacy_empty_ftv_full_field: bool = True


@dataclass
class ModelConfig:
    """Architecture dimensions for the paper model."""

    image_channels: int = 8
    geometry_dim: int = 9
    base_channels: int = 16
    latent_dim: int = 192
    predictor_depth: int = 3
    predictor_heads: int = 4
    predictor_mlp_dim: int = 512
    response_dim: int = 64
    response_hidden_dim: int = 256
    response_depth: int = 1
    response_experts: int = 6
    expert_hidden_dim: int = 128
    expert_gate_hidden_dim: int = 128
    expert_temperature: float = 0.4
    expert_scale: float = 0.1
    expert_init_std: float = 0.005
    response_latent_scale: float = 0.05
    response_target_dim: int = 18
    dropout: float = 0.1
    film_scale: float = 0.1


@dataclass
class LossConfig:
    """Weights used by the pCR-free pretraining objective."""

    prediction: float = 1.0
    sigreg: float = 0.09
    response_score: float = 0.05
    state_delta_contrast: float = 0.02
    update_score: float = 0.02
    response_vector: float = 0.02
    response_vector_update: float = 0.02
    gate_route: float = 0.3
    gate_entropy: float = 0.05
    gate_balance: float = 0.2
    prediction_steps: tuple[float, float, float] = (2.0, 1.0, 0.5)
    response_steps: tuple[float, float, float] = (1.0, 1.0, 0.0)
    update_steps: tuple[float, float] = (1.0, 0.0)
    score_regression: float = 0.1
    score_ranking: float = 1.0
    score_rank_margin: float = 0.1
    update_rank_margin: float = 0.05
    min_rank_target_difference: float = 0.05
    contrast_temperature: float = 0.2
    contrast_target_temperature: float = 0.5
    contrast_condition_penalty: float = 1.0


@dataclass
class TrainConfig:
    """Optimization, split, and checkpoint settings."""

    output_dir: str = "runs/corejepa_clean"
    split_seed: int = 2026
    seed: int = 2026
    batch_size: int = 32
    workers: int = 4
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    sigreg_projections: int = 256
    min_latent_std: float = 0.05
    gpus: tuple[int, ...] = (0, 1)


@dataclass
class ReadoutConfig:
    """Frozen Landmark Readout hyperparameter grid."""

    penalties: tuple[str, ...] = ("l1", "l2")
    c_grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    landmark_weights: tuple[float, float, float] = (2.0, 1.0, 0.5)
    max_iter: int = 5000


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _construct(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    for key, value in list(values.items()):
        if isinstance(getattr(cls(), key, None), tuple) and isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig(
        data=_construct(DataConfig, payload.get("data")),
        model=_construct(ModelConfig, payload.get("model")),
        loss=_construct(LossConfig, payload.get("loss")),
        train=_construct(TrainConfig, payload.get("train")),
        readout=_construct(ReadoutConfig, payload.get("readout")),
    )
