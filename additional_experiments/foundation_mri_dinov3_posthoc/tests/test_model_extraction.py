from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri_dinov3 import extraction  # noqa: E402
from foundation_mri_dinov3 import model as model_module  # noqa: E402
from foundation_mri_dinov3.model import (  # noqa: E402
    DINOV3_BACKBONE_PARAMETERS,
    DINOV3_EMBED_DIM,
    DINOV3_PATCH_TOKEN_COUNT,
    DINOV3_PATCH_TOKEN_START,
    DINOV3_REVISION,
    DINOV3_STATE_ENTRY_COUNT,
    DINOV3_TOKEN_COUNT,
    DINOV3_TOKEN_DIM,
    DINOv3Encoder,
    EncoderAudit,
    MODEL_NAME,
)


def _audit() -> EncoderAudit:
    return EncoderAudit(
        model_name=MODEL_NAME,
        repository_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        revision=DINOV3_REVISION,
        artifacts={"model.safetensors": {"size_bytes": 1, "sha256": "0" * 64}},
        parameter_count=DINOV3_BACKBONE_PARAMETERS,
        state_entry_count=DINOV3_STATE_ENTRY_COUNT,
        representation_dim=DINOV3_EMBED_DIM,
        token_contract="synthetic exact-token contract",
        load_coverage="synthetic strict coverage",
        frozen=True,
        offline=True,
        token_disabled=True,
    )


class _TokenBackbone(nn.Module):
    def __init__(self, *, wrong_shape: bool = False, nonfinite: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.wrong_shape = wrong_shape
        self.nonfinite = nonfinite

    def forward(self, *, pixel_values: torch.Tensor, return_dict: bool):
        assert return_dict is True
        count = DINOV3_TOKEN_COUNT - int(self.wrong_shape)
        tokens = torch.zeros(
            pixel_values.shape[0],
            count,
            DINOV3_TOKEN_DIM,
            device=pixel_values.device,
        )
        tokens[:, 0, :] = 2.0
        tokens[:, 1:5, :] = 1000.0
        if count > DINOV3_PATCH_TOKEN_START:
            tokens[:, DINOV3_PATCH_TOKEN_START:, :] = 3.0
        if self.nonfinite:
            tokens[0, 0, 0] = torch.nan
        return SimpleNamespace(last_hidden_state=tokens)


def _set_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def _write_tiny_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    snapshot = tmp_path / DINOV3_REVISION
    snapshot.mkdir()
    config = {
        "architectures": ["DINOv3ViTModel"],
        "model_type": "dinov3_vit",
        "hidden_size": DINOV3_TOKEN_DIM,
        "image_size": 224,
        "patch_size": 16,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "num_register_tokens": 4,
        "layer_norm_eps": 1e-5,
    }
    preprocessor = {
        "size": {"height": 224, "width": 224},
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_resize": True,
        "do_normalize": True,
    }
    payloads = {
        "config.json": json.dumps(config, sort_keys=True).encode("utf-8"),
        "preprocessor_config.json": json.dumps(preprocessor, sort_keys=True).encode(
            "utf-8"
        ),
        "model.safetensors": b"synthetic safetensors bytes",
    }
    artifacts: dict[str, tuple[int, str]] = {}
    for name, payload in payloads.items():
        (snapshot / name).write_bytes(payload)
        artifacts[name] = (len(payload), hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(model_module, "DINOV3_ARTIFACTS", artifacts)
    return snapshot


def test_encoder_excludes_four_register_tokens_and_concatenates_cls_patch_mean() -> (
    None
):
    encoder = DINOv3Encoder(_TokenBackbone(), _audit())
    with torch.inference_mode():
        representation = encoder(torch.zeros(2, 3, 224, 224))
    assert representation.shape == (2, DINOV3_EMBED_DIM)
    assert torch.equal(representation[:, :DINOV3_TOKEN_DIM], torch.full((2, 768), 2.0))
    assert torch.equal(representation[:, DINOV3_TOKEN_DIM:], torch.full((2, 768), 3.0))
    assert DINOV3_PATCH_TOKEN_START == 5
    assert DINOV3_PATCH_TOKEN_COUNT == 196
    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_encoder_preprocessing_and_finite_shape_gates() -> None:
    encoder = DINOv3Encoder(_TokenBackbone(), _audit())
    normalized = encoder.preprocess(torch.zeros(1, 3, 224, 224))
    expected = (
        torch.tensor([0.5, 0.5, 0.5]) - torch.tensor([0.485, 0.456, 0.406])
    ) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(normalized[0, :, 0, 0], expected)
    with pytest.raises(ValueError, match="224"):
        encoder(torch.zeros(1, 3, 225, 224))
    invalid = torch.zeros(1, 3, 224, 224)
    invalid[0, 0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="input"):
        encoder(invalid)
    with pytest.raises(AssertionError, match="token contract"):
        DINOv3Encoder(_TokenBackbone(wrong_shape=True), _audit())(
            torch.zeros(1, 3, 224, 224)
        )
    with pytest.raises(FloatingPointError, match="tokens"):
        DINOv3Encoder(_TokenBackbone(nonfinite=True), _audit())(
            torch.zeros(1, 3, 224, 224)
        )


def test_offline_runtime_forbids_fallback_and_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="strict offline"):
        model_module.require_offline_runtime()
    _set_offline_environment(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "must-not-be-consumed")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        model_module.require_offline_runtime()
    monkeypatch.delenv("HF_TOKEN")
    model_module.require_offline_runtime()


def test_snapshot_revision_inventory_size_and_sha_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _write_tiny_snapshot(tmp_path, monkeypatch)
    assert model_module.validate_snapshot(snapshot) == snapshot.resolve()
    (snapshot / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        model_module.validate_snapshot(snapshot)
    (snapshot / "unexpected.txt").unlink()
    (snapshot / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        model_module.validate_snapshot(snapshot)
    wrong_revision = tmp_path / "wrong-revision"
    snapshot.rename(wrong_revision)
    with pytest.raises(ValueError, match="revision mismatch"):
        model_module.validate_snapshot(wrong_revision)


def test_loader_passes_only_local_path_and_disables_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _write_tiny_snapshot(tmp_path, monkeypatch)
    _set_offline_environment(monkeypatch)
    monkeypatch.setattr(model_module, "DINOV3_BACKBONE_PARAMETERS", 2)
    monkeypatch.setattr(model_module, "DINOV3_STATE_ENTRY_COUNT", 1)
    observed: dict[str, object] = {}

    class FakeDINOv3ViTModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))

        @classmethod
        def from_pretrained(cls, source: Path, **kwargs):
            observed["source"] = source
            observed.update(kwargs)
            return cls(), {
                "missing_keys": set(),
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            }

        def forward(self, *, pixel_values: torch.Tensor, return_dict: bool):
            tokens = torch.zeros(
                pixel_values.shape[0], DINOV3_TOKEN_COUNT, DINOV3_TOKEN_DIM
            )
            return SimpleNamespace(last_hidden_state=tokens)

    fake_transformers = ModuleType("transformers")
    fake_transformers.DINOv3ViTModel = FakeDINOv3ViTModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    encoder = model_module.load_dinov3_encoder(snapshot)
    assert observed["source"] == snapshot.resolve()
    assert observed["local_files_only"] is True
    assert observed["token"] is False
    assert observed["revision"] == DINOV3_REVISION
    assert observed["use_safetensors"] is True
    assert observed["output_loading_info"] is True
    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    audit = model_module.model_audit(encoder)
    assert audit["parameter_count"] == 2
    assert audit["state_entry_count"] == 1
    assert "snapshot" not in audit and "path" not in audit


def test_patient_extraction_is_4_by_2_by_1536_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEncoder(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            values = images[:, 0, 0, 0].float().view(-1, 1)
            return values.expand(-1, DINOV3_EMBED_DIM)

    def fake_stack(visit: torch.Tensor, axis: str) -> torch.Tensor:
        value = float(visit.reshape(-1)[0]) + (10.0 if axis == "LOCAL" else 0.0)
        return torch.full((3, 3, 224, 224), value)

    monkeypatch.setattr(extraction, "dino_slice_stack", fake_stack)
    image = np.zeros((4, 1, 1, 1, 1), dtype=np.float32)
    image[:, 0, 0, 0, 0] = np.arange(4, dtype=np.float32)
    output = extraction._dinov3_patient(
        image, FakeEncoder(), torch.device("cpu"), "fp32", batch_size=2
    )
    assert output.shape == (4, 2, DINOV3_EMBED_DIM)
    assert output.dtype == np.dtype(np.float32)
    assert np.isfinite(output).all()
    assert np.array_equal(output[:, 0, 0], np.arange(4, dtype=np.float32))
    assert np.array_equal(output[:, 1, 0], np.arange(4, dtype=np.float32) + 10.0)


def test_private_shard_schema_signature_and_mode_are_strict(tmp_path: Path) -> None:
    shard = tmp_path / "shard.private.npz"
    representation = np.zeros((4, 2, DINOV3_EMBED_DIM), dtype=np.float32)
    extraction.atomic_private_npz(
        shard,
        {
            "patient_id": np.asarray("P001"),
            "representation": representation,
            "signature_sha256": np.asarray("a" * 64),
        },
    )
    observed = extraction._validate_shard(
        shard, patient_id="P001", signature_sha256="a" * 64
    )
    assert np.array_equal(observed, representation)
    assert shard.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="another extraction contract"):
        extraction._validate_shard(shard, patient_id="P001", signature_sha256="b" * 64)
    shard.chmod(0o644)
    with pytest.raises(PermissionError, match="permissions"):
        extraction._validate_shard(shard, patient_id="P001", signature_sha256="a" * 64)


def test_smoke_is_resumable_private_and_never_combines_or_loads_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_offline_environment(monkeypatch)
    patient_ids = tuple(f"P{index:03d}" for index in range(808))
    cache = {patient_id: object() for patient_id in patient_ids}
    monkeypatch.setattr(
        extraction,
        "_load_primary_population",
        lambda fold, cache_manifest: (patient_ids, cache),
    )
    calls = {"images": 0}

    def fake_load_dce7(_source: object) -> np.ndarray:
        calls["images"] += 1
        return np.zeros((4, 1, 1, 1, 1), dtype=np.float32)

    monkeypatch.setattr(extraction, "load_dce7", fake_load_dce7)
    monkeypatch.setattr(
        extraction,
        "_dinov3_patient",
        lambda *args, **kwargs: np.zeros((4, 2, DINOV3_EMBED_DIM), dtype=np.float32),
    )
    monkeypatch.setattr(extraction, "load_dinov3_encoder", lambda _: nn.Identity())
    audit = _audit().as_dict()
    monkeypatch.setattr(extraction, "model_audit", lambda _: audit)

    common = dict(
        snapshot_dir=tmp_path / "local-snapshot",
        fold_manifest=tmp_path / "fold.csv",
        cache_manifest=tmp_path / "cache.csv",
        output_root=tmp_path / "features",
        device_name="cpu",
        precision="fp32",
        batch_size=2,
        limit=2,
    )
    assert extraction.extract_features(**common) is None
    assert calls["images"] == 2
    run_root = tmp_path / "features" / "smoke_2" / MODEL_NAME
    shards = sorted((run_root / "shards").glob("*.private.npz"))
    assert len(shards) == 2
    assert all(path.stat().st_mode & 0o077 == 0 for path in shards)
    assert not (run_root / "frozen_features.private.npz").exists()
    assert not (run_root / "execution.private.json").exists()
    contract_text = (run_root / "contract.private.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in contract_text

    # The second run proves matching shards are validated and resumed instead
    # of being overwritten or rereading the image cache.
    monkeypatch.setattr(
        extraction,
        "load_dce7",
        lambda *_: (_ for _ in ()).throw(AssertionError("image reread on resume")),
    )
    assert extraction.extract_features(**common) is None
    assert calls["images"] == 2


def test_formal_requires_cuda_and_refuses_existing_combined_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(extraction.SNAPSHOT_ENV, str(tmp_path / "snapshot"))
    monkeypatch.setattr(
        extraction,
        "_verify_formal_model_input_lock",
        lambda _: {"status": "PASS", "lock_sha256": "a" * 64},
    )
    common = dict(
        snapshot_dir=None,
        fold_manifest=tmp_path / "fold.csv",
        cache_manifest=tmp_path / "cache.csv",
        output_root=tmp_path / "features",
        precision="bf16",
        batch_size=64,
        limit=None,
    )
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        extraction.extract_features(device_name="cpu", **common)

    destination = (
        tmp_path / "features" / "formal" / MODEL_NAME / "frozen_features.private.npz"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"must not be overwritten")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    with pytest.raises(FileExistsError, match="already exists"):
        extraction.extract_features(device_name="cuda:0", **common)
    assert destination.read_bytes() == b"must not be overwritten"


def _load_cli_script():
    path = ROOT / "scripts" / "extract_features.py"
    spec = importlib.util.spec_from_file_location("dinov3_test_extract_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_empty_argv_is_formal_and_nonempty_requires_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_script()
    observed: list[dict[str, object]] = []

    def fake_extract(**kwargs):
        observed.append(kwargs)
        return tmp_path / "private" / "frozen_features.private.npz"

    monkeypatch.setattr(cli, "extract_features", fake_extract)
    assert cli.main([]) == 0
    assert observed[-1]["snapshot_dir"] is None
    assert observed[-1]["limit"] is None
    assert observed[-1]["device_name"] == "cuda:0"
    assert observed[-1]["precision"] == "bf16"
    assert capsys.readouterr().out.strip() == (
        "combined_feature_file=frozen_features.private.npz"
    )
    with pytest.raises(ValueError, match="empty argument vector"):
        cli.main(["--precision", "fp32"])
    assert (
        cli.main(
            [
                "--limit",
                "1",
                "--snapshot",
                str(tmp_path / "snapshot"),
                "--device",
                "cpu",
                "--precision",
                "fp32",
            ]
        )
        == 0
    )
    assert observed[-1]["limit"] == 1
    assert observed[-1]["snapshot_dir"] == tmp_path / "snapshot"


def test_formal_snapshot_is_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(extraction.SNAPSHOT_ENV, "/private/local/revision")
    resolved = extraction.resolve_snapshot_dir(None, formal=True)
    assert resolved == Path("/private/local/revision")
    with pytest.raises(ValueError, match="env-only"):
        extraction.resolve_snapshot_dir(Path("/tmp/override"), formal=True)
    monkeypatch.delenv(extraction.SNAPSHOT_ENV)
    with pytest.raises(RuntimeError, match=extraction.SNAPSHOT_ENV):
        extraction.resolve_snapshot_dir(None, formal=True)
