from pathlib import Path
import importlib.util
import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("static_dce", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_expanded_crop_shape_and_mask_alignment():
    image = np.zeros((20, 30, 10, 3), dtype=np.float32)
    mask = np.zeros((20, 30, 10), dtype=np.float32)
    image[5:10, 8:16, 2:7] = 1
    mask[5:10, 8:16, 2:7] = 1
    bbox = module._bbox_from_mask(mask)
    x = module._expanded_crop(image, *bbox, order=1)
    m = module._expanded_crop(mask, *bbox, order=0)
    assert x.shape == (3, 32, 96, 96)
    assert m.shape == (32, 96, 96)
    assert m.sum() > 0


def test_youden_uses_probabilities():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = module.youden(y, p)
    assert 0.4 < threshold <= 0.6
