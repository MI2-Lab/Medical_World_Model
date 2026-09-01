import numpy as np
import torch

from dinov3_rg.contracts import ARMS, FOLDS, SEEDS, SUMMARY_SHAPE, load_protocol
from dinov3_rg.extraction import central_local_slices, prepare_grayscale_slices, summarize_tokens


def test_protocol_matrix_and_summary_contract():
    protocol = load_protocol()
    assert tuple(protocol["training"]["seeds"]) == SEEDS
    assert tuple(protocol["training"]["folds"]) == FOLDS
    assert tuple(protocol["training"]["arms"]) == ARMS
    assert SUMMARY_SHAPE == (4, 7, 32, 2304)


def test_central_local_crop_is_fixed_and_centered():
    image = np.arange(4 * 7 * 112 * 176 * 160, dtype=np.float32).reshape(4, 7, 112, 176, 160)
    local = central_local_slices(image)
    assert local.shape == (4, 7, 32, 72, 72)
    assert np.array_equal(local, image[:, :, 40:72, 52:124, 44:116])


def test_grayscale_replication_and_token_register_exclusion():
    pixels = prepare_grayscale_slices(torch.zeros(2, 72, 72))
    assert pixels.shape == (2, 3, 224, 224)
    hidden = torch.zeros(2, 201, 768)
    hidden[:, 0] = 3.0
    hidden[:, 1:5] = 9999.0
    hidden[:, 5:] = torch.arange(196, dtype=torch.float32).view(1, 196, 1)
    summary = summarize_tokens(hidden)
    assert summary.shape == (2, 2304)
    assert torch.all(summary[:, :768] == 3.0)
    assert torch.allclose(summary[:, 768:1536], torch.full((2, 768), 97.5))
    assert float(summary.max()) < 9999.0
