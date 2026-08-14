"""Image-only adapter for the confirmed LOCAL3 response-state checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import ARMS, PRIMARY_TIMINGS, InputPaths, validate_seed_fold


STATE_DIM = 192
PROJECTION_HIDDEN_DIM = 128
PROJECTION_DIM = 64
VISITS = 4
IMAGE_CHANNELS = 7
IMAGE_SHAPE_ZYX = (112, 176, 160)
LOCAL3_SOURCE_SHA256 = {
    "__init__.py": "0958d0c6530e249d2da6fa27ff1866ba717ccab31eea8c1402acac686d11504b",
    "contracts.py": "489694ce63b3c699578250f117764712a219ae45371dbd1beb67c2a06087437a",
    "model.py": "976b3c0121dbbb5cf00af679d0cff79c5b2a48b71a118ddb6183c3fcb7194984",
    "pooling.py": "52ef08c85ed256a46c686c7f1afd1b66219735508f4a29cc0922fdb6096bb25c",
    "upstream.py": "8e537d514eabdd7f7c1d7c2234d6fc221610ea0ab5791e9e628e58c6f0e7a4de",
}
UPSTREAM_SOURCE_SHA256 = {
    "dgrs/__init__.py": "c18fa03739e604a77018975ec1d2e7ed00339d8b6a529562446c845e9200b9b8",
    "dgrs/config.py": "4460ce3413e2cb936a6fd3cbb7f16224af3af286b6784933688cd12d0ec47516",
    "dgrs/data.py": "15b4b68ad45c935e313b893b0ce849877311c98d6c5c0c45495e8e9200240943",
    "dgrs/model.py": "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9",
    "dgrs/targets.py": "28fbf66f93c8541dfa5ecc7ebcf65d4143a9a605b3ce98be48355d5ab679ffac",
    "dgrs/training.py": "76f9108df0ca8c0ff69e514cff3bab1d5e316d946da60c5f530dd7b9706d3815",
    "c1b_spatial_audit/__init__.py": "55dbb7a79f6248075464cb983617296bc71cd391b0c08d956c077f3ce0c75584",
    "c1b_spatial_audit/pooling.py": "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ImportError(f"{label} is missing from the confirmed source root")
    observed = _file_sha256(path)
    if observed != expected:
        raise ImportError(
            f"{label} source hash drifted: expected {expected}, observed {observed}"
        )


def verify_confirmed_source_root(root: str | Path) -> dict[str, str]:
    """Verify the confirmation package and every transitive source dependency."""

    confirmation_root = Path(root).expanduser().resolve(strict=True)
    package = confirmation_root / "src" / "lg_response_pilot"
    observed: dict[str, str] = {}
    for relative, expected in LOCAL3_SOURCE_SHA256.items():
        path = package / relative
        _verify_hash(path, expected, f"confirmed LOCAL3 {relative}")
        observed[f"lg_response_pilot/{relative}"] = expected

    # The confirmation package binds these dependencies by repository-relative
    # paths. Derive the authoritative private repo from the confirmation root.
    try:
        private_root = confirmation_root.parents[1]
    except IndexError as exc:  # pragma: no cover - guarded by path structure below.
        raise ImportError("confirmed LOCAL3 root has invalid repository placement") from exc
    expected_location = (
        private_root / "additional_experiments" / "local_response_state_multiseed_confirmation"
    ).resolve()
    if confirmation_root != expected_location:
        raise ImportError(
            "confirmed LOCAL3 must resolve beneath the authoritative private repository root"
        )
    upstream_roots = {
        "dgrs": private_root
        / "additional_experiments"
        / "g3_multiseed_generalization"
        / "src",
        "c1b_spatial_audit": private_root
        / "additional_experiments"
        / "c1b_spatial_pooling_bottleneck_audit"
        / "src",
    }
    for relative, expected in UPSTREAM_SOURCE_SHA256.items():
        namespace = relative.split("/", 1)[0]
        path = upstream_roots[namespace] / relative
        _verify_hash(path, expected, f"confirmed LOCAL3 dependency {relative}")
        observed[relative] = expected
    return observed


def _purge_package(name: str) -> None:
    for key in tuple(sys.modules):
        if key == name or key.startswith(f"{name}."):
            del sys.modules[key]


def _load_confirmed_package(root: Path) -> ModuleType:
    verify_confirmed_source_root(root)
    package_dir = root / "src" / "lg_response_pilot"
    init_path = package_dir / "__init__.py"
    # A repository may already have imported a same-named package from another
    # experiment. Remove it and load this exact verified path explicitly.
    _purge_package("lg_response_pilot")
    specification = importlib.util.spec_from_file_location(
        "lg_response_pilot",
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if specification is None or specification.loader is None:
        raise ImportError("could not construct confirmed LOCAL3 import specification")
    package = importlib.util.module_from_spec(specification)
    sys.modules["lg_response_pilot"] = package
    specification.loader.exec_module(package)
    model_path = (package_dir / "model.py").resolve()
    module = __import__("lg_response_pilot.model", fromlist=["model"])
    if Path(inspect.getfile(module.LocalGlobalResponseWorldModel)).resolve() != model_path:
        raise ImportError("LOCAL3 class did not resolve to the confirmed source root")
    return module


@dataclass(frozen=True)
class Local3Checkpoint:
    """Verified confirmed backbone plus aggregate provenance."""

    model: nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    seed: int
    fold: int
    epoch: int
    source_hashes: Mapping[str, str]


def load_confirmed_local3(
    checkpoint_path: str | Path,
    confirmation_root: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_seed: int | None = None,
    expected_fold: int | None = None,
) -> Local3Checkpoint:
    """Strictly load one real selected/test-blind LOCAL3 checkpoint."""

    root = Path(confirmation_root).expanduser().resolve(strict=True)
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("LOCAL3 checkpoint is outside the confirmed source root") from exc
    parts = relative.parts
    if len(parts) != 6 or parts[:2] != ("checkpoints", "formal_4x8"):
        raise ValueError("LOCAL3 checkpoint path does not match the formal matrix contract")
    if parts[2] not in {f"seed_{value}" for value in (2026, 3026, 4026)}:
        raise ValueError("LOCAL3 checkpoint path has an unregistered seed")
    if parts[3] != "LOCAL3" or parts[4] not in {f"fold_{value}" for value in range(5)}:
        raise ValueError("LOCAL3 checkpoint path has the wrong arm or fold")
    # Pathlib parts include no empty component; formal layout ends at fold_X/selected.pt.
    if parts[-1] != "selected.pt":
        raise ValueError("LOCAL3 checkpoint must be the selected.pt artifact")
    path_seed = int(parts[2].removeprefix("seed_"))
    path_fold = int(parts[4].removeprefix("fold_"))
    validate_seed_fold(path_seed, path_fold)
    if expected_seed is not None and int(expected_seed) != path_seed:
        raise ValueError("checkpoint path seed disagrees with requested seed")
    if expected_fold is not None and int(expected_fold) != path_fold:
        raise ValueError("checkpoint path fold disagrees with requested fold")

    source_hashes = verify_confirmed_source_root(root)
    module = _load_confirmed_package(root)
    model, payload = module.load_checkpoint_for_evaluation(source, device=device)
    if not isinstance(payload, Mapping):
        raise ValueError("confirmed LOCAL3 loader returned invalid checkpoint metadata")
    checks = {
        "schema_version": payload.get("schema_version") == 1,
        "stage": payload.get("stage") == "local_response_state_multiseed_confirmation",
        "arm": payload.get("arm") == "LOCAL3",
        "architecture": payload.get("architecture") == "LOCAL",
        "input_kind": payload.get("input_kind") == "c1b",
        "grounded": payload.get("grounded") is True,
        "selected": payload.get("selected") is True,
        "test_blind": payload.get("test_data_used") is False,
        "pcr_free_source": payload.get("pcr_used") is False,
        "no_delta_ftv": payload.get("delta_ftv_used") is False,
        "seed": payload.get("seed_base") == path_seed,
        "fold": payload.get("fold") == path_fold,
        "preregistered": payload.get("preregistration_status") == "PASS",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"confirmed LOCAL3 checkpoint contract failed: {failed}")
    contract = payload.get("architecture_contract")
    if not isinstance(contract, Mapping) or any(
        (
            contract.get("arm") != "LOCAL3",
            contract.get("architecture") != "LOCAL",
            contract.get("image_channels") != IMAGE_CHANNELS,
            contract.get("input_shape_zyx") != list(IMAGE_SHAPE_ZYX),
            contract.get("response_dim") != STATE_DIM,
            contract.get("response_projection") != "Linear(128,192)+LayerNorm(192)",
            contract.get("roi_mask_use") != "absent",
            contract.get("ftv_is_forward_input") is not False,
        )
    ):
        raise ValueError("confirmed LOCAL3 architecture contract is incompatible")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get("arm") != "LOCAL3":
        raise ValueError("confirmed LOCAL3 model configuration is incompatible")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("confirmed LOCAL3 checkpoint epoch is invalid")
    return Local3Checkpoint(
        model=model,
        checkpoint_path=source,
        checkpoint_sha256=_file_sha256(source),
        seed=path_seed,
        fold=path_fold,
        epoch=epoch,
        source_hashes=source_hashes,
    )


def load_local3_for_cell(
    paths: InputPaths,
    seed: int,
    fold: int,
    *,
    device: str | torch.device = "cpu",
) -> Local3Checkpoint:
    seed_value, fold_value = validate_seed_fold(seed, fold)
    return load_confirmed_local3(
        paths.checkpoint_path(seed_value, fold_value),
        paths.confirmation_root,
        device=device,
        expected_seed=seed_value,
        expected_fold=fold_value,
    )


class ContrastiveProjectionHead(nn.Sequential):
    """The frozen 192 -> 128 -> 64 nonlinear projection contract."""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(STATE_DIM, PROJECTION_HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(PROJECTION_HIDDEN_DIM),
            nn.Linear(PROJECTION_HIDDEN_DIM, PROJECTION_DIM),
        )


@dataclass(frozen=True)
class ConditionalCeilingOutput:
    response_state: torch.Tensor
    projected_state: torch.Tensor
    pcr_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[()]


class ConditionalCeilingModel(nn.Module):
    """Confirmed LOCAL3 plus image-derived supervised ceiling heads.

    The forward signature contains one argument, ``image``. Clinical subtype,
    treatment arm, FTV, ROI masks, and labels are not representable as model
    inputs; they remain outside the module and are used only by loss/sampling
    orchestration.
    """

    def __init__(self, backbone: nn.Module, arm: str) -> None:
        super().__init__()
        name = str(arm).upper()
        if name not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        required = ("encoder", "response_projection", "encode_response")
        if any(not hasattr(backbone, value) for value in required):
            raise TypeError("backbone does not implement the confirmed LOCAL3 response contract")
        self.arm = name
        self.backbone = backbone
        self.projection_head = ContrastiveProjectionHead()
        # Timing-specific linear heads consume literal projected-state prefixes:
        # T0=64, T0-T1=128, T0-T2=192. They are training/validation only.
        self.pcr_heads = nn.ModuleList(
            [nn.Linear(PROJECTION_DIM * visits, 1) for visits in (1, 2, 3)]
        )
        self.configure_trainability()

    def configure_trainability(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.arm == "B0":
            return
        self.projection_head.requires_grad_(True)
        if self.arm == "B1":
            return
        self.pcr_heads.requires_grad_(True)
        self.backbone.response_projection.requires_grad_(True)
        if self.arm == "B2":
            features = getattr(self.backbone.encoder, "features", None)
            if not isinstance(features, nn.Sequential) or len(features) != 4:
                raise ValueError("B2 requires the confirmed four-stage encoder")
            features[3].requires_grad_(True)
        elif self.arm == "B3":
            self.backbone.encoder.requires_grad_(True)

    def encode_response(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.ndim != 6:
            raise ValueError("image must be [B,V,7,Z,Y,X]")
        if int(image.shape[1]) != VISITS or int(image.shape[2]) != IMAGE_CHANNELS:
            raise ValueError("image must contain exactly T0-T3 and seven DCE channels")
        if tuple(int(value) for value in image.shape[-3:]) != IMAGE_SHAPE_ZYX:
            raise ValueError(f"image geometry must be frozen C1B-H {IMAGE_SHAPE_ZYX}")
        response = self.backbone.encode_response(image)
        expected = (int(image.shape[0]), VISITS, STATE_DIM)
        if tuple(response.shape) != expected:
            raise ValueError(f"LOCAL3 response must have shape {expected}")
        return response

    def project_response(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 3 or response.shape[-1] != STATE_DIM:
            raise ValueError("response must be [B,V,192]")
        return self.projection_head(response)

    def timing_logits(self, projected: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.arm in {"B0", "B1"}:
            return ()
        if projected.ndim != 3 or projected.shape[1] < 3 or projected.shape[2] != PROJECTION_DIM:
            raise ValueError("projected state must be [B,>=3,64]")
        return tuple(
            head(projected[:, : index + 1].reshape(projected.shape[0], -1)).squeeze(-1)
            for index, head in enumerate(self.pcr_heads)
        )

    def forward(self, image: torch.Tensor) -> ConditionalCeilingOutput:
        response = self.encode_response(image)
        projected = self.project_response(response)
        logits = self.timing_logits(projected)
        return ConditionalCeilingOutput(response, projected, logits)  # type: ignore[arg-type]

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "arm": self.arm,
            "backbone": "confirmed LOCAL3",
            "backbone_response": "online pre-projector r",
            "state_dim": STATE_DIM,
            "projection": "Linear(192,128)-GELU-LayerNorm(128)-Linear(128,64)",
            "timings": list(PRIMARY_TIMINGS),
            "pcr_heads": ["Linear(64,1)", "Linear(128,1)", "Linear(192,1)"],
            "bce_weight": 0.25 if self.arm in {"B2", "B3"} else 0.0,
            "forward_inputs": ["image"],
            "clinical_inputs": False,
            "treatment_input": False,
            "ftv_input": False,
            "roi_mask_input": False,
        }


def build_ceiling_model(
    arm: str,
    backbone: nn.Module | Local3Checkpoint,
    *,
    head_seed: int | None = None,
) -> ConditionalCeilingModel:
    """Build an arm without perturbing the caller's RNG stream."""

    module = backbone.model if isinstance(backbone, Local3Checkpoint) else backbone
    seed = 0 if head_seed is None else int(head_seed)
    if isinstance(head_seed, bool):
        raise ValueError("head_seed must be an integer")
    devices: list[int] = []
    try:
        first_parameter = next(module.parameters())
    except StopIteration:
        first_parameter = None
    if first_parameter is not None and first_parameter.is_cuda:
        devices = [first_parameter.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = ConditionalCeilingModel(module, arm)
    validate_model_contract(model)
    return model


def load_ceiling_model(
    arm: str,
    checkpoint_path: str | Path,
    confirmation_root: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_seed: int | None = None,
    expected_fold: int | None = None,
    head_seed: int | None = None,
) -> tuple[ConditionalCeilingModel, Local3Checkpoint]:
    checkpoint = load_confirmed_local3(
        checkpoint_path,
        confirmation_root,
        device=device,
        expected_seed=expected_seed,
        expected_fold=expected_fold,
    )
    model = build_ceiling_model(arm, checkpoint, head_seed=head_seed).to(device)
    return model, checkpoint


def validate_model_contract(model: ConditionalCeilingModel) -> None:
    if not isinstance(model, ConditionalCeilingModel):
        raise TypeError("model must be ConditionalCeilingModel")
    if len(model.projection_head) != 4:
        raise ValueError("projection head layer count drifted")
    linear1, activation, normalization, linear2 = model.projection_head
    if not isinstance(linear1, nn.Linear) or (
        linear1.in_features,
        linear1.out_features,
    ) != (STATE_DIM, PROJECTION_HIDDEN_DIM):
        raise ValueError("projection head must begin Linear(192,128)")
    if not isinstance(activation, nn.GELU):
        raise ValueError("projection head must use GELU")
    if not isinstance(normalization, nn.LayerNorm) or tuple(normalization.normalized_shape) != (
        PROJECTION_HIDDEN_DIM,
    ):
        raise ValueError("projection head must use LayerNorm(128)")
    if not isinstance(linear2, nn.Linear) or (
        linear2.in_features,
        linear2.out_features,
    ) != (PROJECTION_HIDDEN_DIM, PROJECTION_DIM):
        raise ValueError("projection head must end Linear(128,64)")
    dimensions = [(head.in_features, head.out_features) for head in model.pcr_heads]
    if dimensions != [(64, 1), (128, 1), (192, 1)]:
        raise ValueError("timing-specific pCR head dimensions drifted")

    trainable = set(model.trainable_parameter_names())
    projection = {
        name for name, _ in model.named_parameters() if name.startswith("projection_head.")
    }
    pcr = {name for name, _ in model.named_parameters() if name.startswith("pcr_heads.")}
    response = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("backbone.response_projection.")
    }
    encoder = {
        name for name, _ in model.named_parameters() if name.startswith("backbone.encoder.")
    }
    final_stage = {
        name
        for name in encoder
        if name.startswith("backbone.encoder.features.3.")
    }
    expected = {
        "B0": set(),
        "B1": projection,
        "B2": projection | pcr | response | final_stage,
        "B3": projection | pcr | response | encoder,
    }[model.arm]
    if trainable != expected:
        raise ValueError(
            f"{model.arm} trainability drifted; unexpected={sorted(trainable - expected)}, "
            f"missing={sorted(expected - trainable)}"
        )
    signature = inspect.signature(model.forward)
    if tuple(signature.parameters) != ("image",):
        raise ValueError("model forward must accept only the image tensor")


# Compatibility aliases for scripts that use the adapter terminology.
build_model = build_ceiling_model
load_local3_checkpoint = load_confirmed_local3


__all__ = [
    "ConditionalCeilingModel",
    "ConditionalCeilingOutput",
    "ContrastiveProjectionHead",
    "Local3Checkpoint",
    "build_ceiling_model",
    "build_model",
    "load_ceiling_model",
    "load_confirmed_local3",
    "load_local3_checkpoint",
    "load_local3_for_cell",
    "validate_model_contract",
    "verify_confirmed_source_root",
]
